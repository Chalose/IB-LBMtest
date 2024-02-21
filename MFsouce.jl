# 返回流体作用力Fp，力矩Tp；注入力的x、y分量；计算体积力源项Fsource

using SparseArrayKit, Interpolations, Base.Threads, SparseArrays

function MultiFsourceSolver!(XcYcK::Matrix, XLYL::Matrix,
    δx, δt, NumL::Int, k::Int, NumP::Int,
    upK::Matrix, ωpK::Vector,
    Cls, ω, cₛ, τf,
    u::Matrix, v::Matrix, ρ₀::Float64, f::Array{Float64,3},
    Fsource::SparseArray, VForceX::SparseMatrixCSC, VForceY::SparseMatrixCSC,
    Force::Array{Float64,3}, Torque::Matrix)
    #=
    XcYcK 为当前时间层k所有粒子的质心坐标
    XLYL 存有所有的L节点质心系坐标
    NumL 为单个粒子L节点总数
    k 为当前时间层序数
    NumP 为沉降的粒子总数
    upK、ωpK 为当前时间层k的所有粒子的质心速度、角速度
    Force、Torque 存储流体对各粒子的合力、合力矩
    Fsource、VForceX、VForceY输入本函数前归零
    =#

    # 某节点第α速度层的力密度矢量
    function gLα(f_αk::Float64, α::Int, ub::Vector)
        g_αk = -2 / δt * (f_αk - ω[α] * ρ₀ / cₛ^2 * Cls[α]' * ub) * Cls[α]

        return g_αk[1], g_αk[2]
    end
    # 伪Delta函数
    function Dfun(xi::Int, yi::Int, Xb::Float64, Yb::Float64)
        a = xi - Xb
        b = yi - Yb
        Value = 0.0
        if abs(a) <= 2 && abs(b) <= 2
            Value = 1 / 16 * (1 + cos(π * a / 2)) * (1 + cos(π * b / 2))
        end

        return Value
    end
    # 某节点注入力矢量计算
    function VForce(xi::Int, yi::Int, Xb::Float64, Yb::Float64, gLx::Float64, gLy::Float64)
        Fx = Dfun(xi, yi, Xb, Yb) * 1 / NumL * gLx
        Fy = Dfun(xi, yi, Xb, Yb) * 1 / NumL * gLy

        return Fx, Fy
    end
    # 某节点第α速度层力源项计算
    function Fα(xi::Int, yi::Int, α::Int, Fn::Vector)
        Fα_n = (1 - 1 / (2 * τf)) * ω[α] * Fn' * (
                   1 / cₛ^2 * (Cls[α] - [u[yi, xi], v[yi, xi]]) + 1 / cₛ^4 * (Cls[α]' * [u[yi, xi], v[yi, xi]]) * Cls[α]
               )

        return Fα_n
    end
    ###############################################################################################################
    ij_XL = zeros(Int, 2, NumL, NumP)  # 存放不同粒子的所有L节点的最近邻流体格点行列索引(确定插值中心)
    for n in 1:NumP
        for m in 1:NumL
            ij_XL[1, m, n] = Int(round(XcYcK[2, n] + XLYL[2, m]))  # 行索引
            ij_XL[2, m, n] = Int(round(XcYcK[1, n] + XLYL[1, m]))  # 列索引
        end
    end
    fsource = zeros(size(Fsource))  # 初始化求解力源项(稀疏数组涉及字典dict，不可直接并发)
    vforceX = zeros(size(VForceX))  # 初始化求解体积力x分量
    vforceY = zeros(size(VForceY))  # 初始化求解体积力y分量
    Fp = zeros(2, NumP)  # 存储各粒子受的合力
    Tp = zeros(1, NumP)  # 存储各粒子受的合力矩
    
    @sync for n in 1:NumP
        Threads.@spawn begin
            ub = [0.0, 0.0]  # 初始化单个粒子各节点的绝对速度
            Vau_gL = zeros(2, NumL)  # 刷新单个粒子各L节点的恢复力密度
            fp = [0.0, 0.0]  # 刷新单个粒子合力
            tp = 0.0  # 刷新单个粒子合力矩
            for m in 1:NumL
                # 第m个节点的绝对坐标
                Xb = XcYcK[1, n] + XLYL[1, m]
                Yb = XcYcK[2, n] + XLYL[2, m]
                # 第m个节点的分布插值样本点
                XLi = ij_XL[1, m, n]
                XLj = ij_XL[2, m, n]
                is = XLi-1:XLi+1  # 行 <=> y
                js = XLj-1:XLj+1  # 列 <=> x
                # 第m个节点的绝对速度
                ub = upK[:, n] + ωpK[n] * [-XLYL[2, m], XLYL[1, m]]
                # 初始化节点受力
                Fn = [0.0, 0.0]
    
                # 计算第n粒子的各节点恢复力密度gL，存放于Vau_gL................................................................
                for α in 1:9
                    fs = [f[i, j, α] for i in is, j in js]  # 插值样本点值
                    cubic_itp = cubic_spline_interpolation((js, is), fs)
                    f_αk = cubic_itp(Xb, Yb)  # 节点分布
    
                    Vau_gL[1, m], Vau_gL[2, m] = (Vau_gL[1, m], Vau_gL[2, m]) .+ gLα(f_αk, α, ub)  # 第k节点恢复力密度
                end
    
                # 计算第n粒子的所受的合力Fp，合力矩Tp..........................................................................
                Fn = -δx^3 / NumL * [Vau_gL[1, m], Vau_gL[2, m]]
                fp = fp + Fn
                tp = tp + XLYL[1, m] * Fn[2] - XLYL[2, m] * Fn[1]
    
                # 计算边界注入体积力VForceX、VForceY、力源项Fsource............................................................
                # 计算该L节点周围的边界力注入格点(21个)
                iE = [XLi + 2, XLi + 2, XLi + 2, XLi + 1, XLi + 1, XLi + 1, XLi + 1, XLi + 1, XLi, XLi, XLi, XLi, XLi, XLi - 1, XLi - 1, XLi - 1, XLi - 1, XLi - 1, XLi - 2, XLi - 2, XLi - 2]
                jE = [XLj - 1, XLj, XLj + 1, XLj - 2, XLj - 1, XLj, XLj + 1, XLj + 2, XLj - 2, XLj - 1, XLj, XLj + 1, XLj + 2, XLj - 2, XLj - 1, XLj, XLj + 1, XLj + 2, XLj - 1, XLj, XLj + 1]
    
                for s in 1:21
                    vforceX[iE[s], jE[s]], vforceY[iE[s], jE[s]] = (vforceX[iE[s], jE[s]], vforceY[iE[s], jE[s]]) .+
                                                                    VForce(jE[s], iE[s], Xb, Yb, Vau_gL[1, m], Vau_gL[2, m])
                    for α in 1:9
                        fsource[iE[s], jE[s], α] = fsource[iE[s], jE[s], α] + Fα(jE[s], iE[s], α, [vforceX[iE[s], jE[s]], vforceY[iE[s], jE[s]]])
                    end
                end
            end
            # 存储第n粒子的合力、合力矩........................................................................................
            Fp[:, n] = fp
            Tp[n] = tp
        end
    end

    # 数据反馈
    Fsource .= SparseArrayKit.SparseArray(fsource)
    VForceX .= sparse(vforceX)
    VForceY .= sparse(vforceY)
    for n in 1:NumP
        Force[:, k+1, n] = Fp[:, n]
        Torque[n, k+1] = Tp[n]
    end
end