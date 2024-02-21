#= 利用浸没边界法计算多个粒子(粒子数NumP >= 2)的沉降过程
D2Q9模型:
7---3---6
| \ | / |
4---1---2
| / | \ |
8---5---9
=#
using Plots, CairoMakie, ProgressBars, SparseArrayKit, SparseArrays
using Base.Threads
Plots.theme(:dark)
println("当前线程总数: ", Threads.nthreads())
include("MFsouce.jl")
include("MInteractForce.jl")
include("MTrackInformation.jl")

function main()
    # 环境参数(各参数均为格子单位)========================================================================= 
    Nx = 150  # 横向格点数
    Ny = 800  # 纵向格点数
    tmax = 8000  # 最大求解时间
    νf = 0.1  # 流体运动粘度
    ρ₀ = 1.0  # 流体初始密度
    Θ = π / 4  # 容器倾角
    aEx = 0.0002  # 外场加速度模
    g = aEx * [cos(Θ), -sin(Θ)]  # 外力场加速度

    # 粒子属性(格子单位)==================================================================================
    dp = 12  # 粒子粒径
    rp = dp / 2
    rhoP = 2.2 * ρ₀  # 粒子密度
    Mp = π * rp^2 * (rhoP - ρ₀)  # 粒子等效质量
    Ip = 1 / 2 * Mp * rp^2  # 粒子等效转动惯量

    # 粒子初始排布=======================================================================================
    numPx = 4  # x方向粒子数
    numPy = 4  # y方向粒子数
    NumP = numPx * numPy  # 总粒子数
    ΔX = (Nx - 1) / (numPx + 1)
    #ΔY = (Ny - 1) / (numPy + 1)
    ΔY = (Ny - 20) / numPy
    XcYc_0 = zeros(2, 1, NumP)  # 粒子初始质心分布
    #
    n = 1
    for i in 1:numPx
        for j in 1:numPy
            # 均匀释放
            XcYc_0[:, 1, n] = [i * ΔX, Ny - 20 - (j - 1) * ΔY]
            # 顶端释放
            #XcYc_0[:, 1, n] = [i * ΔX, Ny - j * ΔY]
            n += 1
        end
    end

    # 离散设置===========================================================================================
    δx = 1.0  # 空间步长
    δt = 1.0  # 时间步长
    NumT = length(0:δt:tmax)  # 总时间步数
    NumL = 80  # 边界Lagrange节点数

    # D2Q9模型===========================================================================================
    c₀ = δx / δt
    cₛ = c₀ / sqrt(3)  # 格子声速
    τf = νf / (cₛ^2 * δt) + 1 / 2  # 流体弛豫时间
    ω = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]  # 速度权重系数
    cx = c₀ * [0 1 0 -1 0 1 -1 -1 1]
    cy = c₀ * [0 0 1 0 -1 1 1 -1 -1]
    # D2Q9离散速度矢量集
    Cls = Array{Float64}[]
    for i in 1:9
        push!(Cls, [cx[i], cy[i]])
    end
    # 平衡态函数
    function fEq(u, v, ρ, Q)
        feq = ω[Q] * ρ .* (1.0 .+ 1 / cₛ^2 * (
            (cx[Q] * u + cy[Q] * v) +
            (cx[Q] * u + cy[Q] * v) .^ 2 / (2 * cₛ^2) -
            (u .^ 2 + v .^ 2) / 2
        ))

        return feq
    end

    # 排斥势参数=========================================================================================
    if NumP <= 25
        A = 25 * Mp * abs(g[2])
    else
        A = NumP * Mp * abs(g[2])
    end
    B = 2.5 * rp

    # 初始化=============================================================================================
    ρ = ones(Ny, Nx) * ρ₀  # 流体密度分布
    u = zeros(Ny, Nx)  # 流体速度分量u、v
    v = zeros(Ny, Nx)
    f = zeros(Ny, Nx, 9)
    # 分布函数初值
    @views for Q in 1:9
        f[:, :, Q] = fEq(u, v, ρ, Q)
    end

    XcYc = zeros(2, NumT, NumP)  # 存储粒子各时间层质心坐标
    for n in 1:NumP
        XcYc[:, 1, n] = XcYc_0[:, 1, n]
    end

    θL = range(0, 2π, NumL)
    XLYL = zeros(2, NumL)  # 边界Lagrange节点的质心系坐标
    XLYL[1, :] = rp * cos.(θL)
    XLYL[2, :] = rp * sin.(θL)

    up = zeros(2, NumT, NumP)  # 各时间层粒子质心速度矢量
    ωp = zeros(NumP, NumT)  # 各时间层粒子角速度
    Force = zeros(2, NumT, NumP)  # 粒子受合力时间序列
    Force[:, 1, :] .= Mp * g  # 初始受力只考虑等效重力
    Torque = zeros(NumP, NumT)  # 粒子受合力矩时间序列

    # 绘图(Makie)
    X_show = range(0, Nx-1)' .* ones(Ny, Nx)
    Y_show = ones(Ny, Nx) .* range(0, Ny-1)
    obst = zeros(Bool, size(X_show))
    for n in 1:NumP
        obst = obst .|| (X_show .- XcYc[1, 1, n]) .^ 2 + (Y_show .- XcYc[2, 1, n]) .^ 2 .<= rp^2
    end
    fig = CairoMakie.Figure(size=(500, 800))
    ax1 = CairoMakie.Axis(fig[1, 1], aspect=Nx/Ny, title="初始粒子分布", xlabel="x", ylabel="y")
    ax2 = CairoMakie.Axis(fig[1, 2], aspect=1, title="Lagrange节点划分")
    CairoMakie.heatmap!(ax1, obst')  # 注意要obst'
    CairoMakie.scatter!(ax2, XLYL[1, :], XLYL[2, :])
    display(fig)

    # 存储设置============================================================================================
    gap = 40  # 每间隔gap个时间步存储一次数据用于返回
    NUM = Int(round(NumT / gap)) + 1
    println(string("总记录帧数NUM = ", NUM))
    save_u = zeros(Ny, Nx, NUM)
    save_v = zeros(Ny, Nx, NUM)
    save_ρ = zeros(Ny, Nx, NUM)
    num = 1

    # 主迭代==============================================================================================
    kSteps = tqdm(1:NumT-1, printing_delay=5)
    for k in kSteps
        # 计算流体对粒子的作用力Fp与力矩Tp，并记入数组Force、Torque中；计算体积力源项Fsource...................
        Fsource = SparseArrayKit.SparseArray(zeros(Ny, Nx, 9))  # SparseArrayKit.jl提供高维数组的稀疏矩阵
        VForceX = sparse(zeros(size(u)))  # 注入体积力的x分布
        VForceY = sparse(zeros(size(u)))  # 注入体积力的y分布
        MultiFsourceSolver!(XcYc[:, k, :], XLYL,
            δx, δt, NumL, k, NumP,
            up[:, k, :], ωp[:, k],
            Cls, ω, cₛ, τf,
            u, v, ρ₀, f,
            Fsource, VForceX, VForceY,
            Force, Torque)
    
        # 计算碰撞步.......................................................................................
        Threads.@threads for Q in 1:9
            f[:, :, Q] = f[:, :, Q] * (1 - 1 / τf) + 1 / τf * fEq(u, v, ρ, Q)
        end
    
        # 计算含力源项迁移步................................................................................
        f = f + Fsource
        @views begin  # 边界迁移存在粒子数不守恒，后期由非平衡外推修正
            f[:, 2:Nx, 2] = f[:, 1:Nx-1, 2]           # 左向右
            f[:, 1:Nx-1, 4] = f[:, 2:Nx, 4]           # 右向左
            f[2:Ny, :, 3] = f[1:Ny-1, :, 3]           # 下向上
            f[1:Ny-1, :, 5] = f[2:Ny, :, 5]           # 上向下
            f[2:Ny, 2:Nx, 6] = f[1:Ny-1, 1:Nx-1, 6]   # 左下向右上
            f[2:Ny, 1:Nx-1, 7] = f[1:Ny-1, 2:Nx, 7]   # 右下向左上
            f[1:Ny-1, 1:Nx-1, 8] = f[2:Ny, 2:Nx, 8]   # 右上向左下
            f[1:Ny-1, 2:Nx, 9] = f[2:Ny, 1:Nx-1, 9]   # 左上向右下
        end
    
        # 宏观量计算........................................................................................
        @views begin
            sum!(ρ, f)
            u = (sum!(u, reshape(cx .* reshape(f, Ny * Nx, 9), Ny, Nx, 9)) + 1 / 2 * VForceX) ./ ρ
            v = (sum!(v, reshape(cy .* reshape(f, Ny * Nx, 9), Ny, Nx, 9)) + 1 / 2 * VForceY) ./ ρ
        end
    
        # 四壁边界非平衡外推.................................................................................
        # 四壁边界宏观量固定
        @views begin
            # 上边界
            u[Ny, :] .= 0.0
            v[Ny, :] .= 0.0
            # 下边界
            u[1, :] .= 0.0
            v[1, :] .= 0.0
            # 左边界
            u[:, 1] .= 0.0
            v[:, 1] .= 0.0
            # 右边界
            u[:, Nx] .= 0.0
            v[:, Nx] .= 0.0
        end
        # 非平衡外推
        # 上边界
        @views for Q in [8, 5, 9]
            f[Ny, :, Q] = fEq(u[Ny, :], v[Ny, :], ρ[Ny, :], Q) + f[Ny-1, :, Q] - fEq(u[Ny-1, :], v[Ny-1, :], ρ[Ny-1, :], Q)
        end
        # 下边界
        @views for Q in [7, 3, 6]
            f[1, :, Q] = fEq(u[1, :], v[1, :], ρ[1, :], Q) + f[2, :, Q] - fEq(u[2, :], v[2, :], ρ[2, :], Q)
        end
        # 左边界
        @views for Q in [6, 2, 9]
            f[:, 1, Q] = fEq(u[:, 1], v[:, 1], ρ[:, 1], Q) + f[:, 2, Q] - fEq(u[:, 2], v[:, 2], ρ[:, 2], Q)
        end
        # 右边界
        @views for Q in [7, 4, 8]
            f[:, Nx, Q] = fEq(u[:, Nx], v[:, Nx], ρ[:, Nx], Q) + f[:, Nx-1, Q] - fEq(u[:, Nx-1], v[:, Nx-1], ρ[:, Nx-1], Q)
        end
    
        # 计算颗粒全部受力(流体作用力，外场驱动力，排斥力)....................................................
        ForceSolver!(NumP, k, Force, XcYc[:, k, :], A, B, Nx, Ny, Mp, g)
    
        # 解算所有粒子新运动信息............................................................................
        TrackSolver!(ωp, up, XcYc, Force, Torque, k, δt, Mp, Ip)
    
        # 后处理用信息......................................................................................
        if k == (num - 1) * gap + 1
            @views save_u[:, :, num] = u
            @views save_v[:, :, num] = v
            @views save_ρ[:, :, num] = ρ
            num += 1
        end
    end

    return save_u, save_v, save_ρ, XcYc, up, NUM, Nx, Ny, gap, rp
end
u, v, ρ, XcYc, up, NUM, Nx, Ny, gap, rp = main();

# 绘图测试======================================================================================================
uv = zeros(Ny, Nx, NUM)
Ω = zeros(Ny, Nx, NUM)
time = 0:gap:(NUM - 1) * gap
# 速度处理函数..................................................................................................
function Velocity!(u, v, uv, NUM, Nx, Ny, XcYc, rp, time)
    # 流场速度模
    for m in 1:NUM 
        uv[:, :, m] = sqrt.(u[:, :, m] .^ 2 + v[:, :, m] .^ 2)
    end

    # 标记粒子
    X = range(0, Nx-1)' .* ones(Ny, Nx)
    Y = ones(Ny, Nx) .* range(0, Ny-1)
    NumP = size(XcYc, 3)
    for m in 1:NUM
        tIndex = time[m] + 1
        obst = zeros(Bool, size(X))
        for n in 1:NumP
            obst = obst .|| (X .- XcYc[1, tIndex, n]) .^ 2 + (Y .- XcYc[2, tIndex, n]) .^ 2 .<= rp^2
        end
        index_obst = findall(obst)
        uv[index_obst, m] .= 0.0
    end
    #
end

# 涡量场计算函数.................................................................................................
function VorticityCal!(
    Nx::Int, Ny::Int, NUM::Int, 
    u::Array{Float64, 3}, v::Array{Float64, 3}, Ω::Array{Float64, 3},
    )
    for num in 1:NUM
        # 中心格点
        for i in 2:Ny-1
            for j in 2:Nx-1
                Ω[i, j, num] = 1 / 2 * ((v[i, j+1, num] - v[i, j-1, num]) - (u[i+1, j, num] - u[i-1, j, num]))
            end
        end
        # 左右边界
        for i in 2:Ny-1
            Ω[i, 1, num] = 1 / 2 * ((-3*v[i, 1, num] + 4*v[i, 2, num] - v[i, 3, num]) - (u[i+1, 1, num] - u[i-1, 1, num]))  # 左
            Ω[i, Nx, num] = 1 / 2 * ((3*v[i, Nx, num] - 4*v[i, Nx-1, num] + v[i, Nx-2, num]) - (u[i+1, Nx, num] - u[i-1, Nx, num]))  # 右
        end
        # 上下边界
        for j in 2:Nx-1
            Ω[Ny, j, num] = 1 / 2 * ((v[Ny, j+1, num] - v[Ny, j-1, num]) - (3*u[Ny, j, num] - 4*u[Ny-1, j, num] + u[Ny-2, j, num]))  # 上
            Ω[1, j, num] = 1 / 2 * ((v[1, j+1, num] - v[1, j-1, num]) - (-3*u[1, j, num] + 4*u[2, j, num] - u[3, j, num]))  # 下
        end
    end
end

Velocity!(u, v, uv, NUM, Nx, Ny, XcYc, rp, time)
VorticityCal!(Nx, Ny, NUM, u, v, Ω)
# 动态图
begin
    pressure = 1/3 * ρ  # 压强场P = cₛ² * ρ
    anime = @animate for num in 1:NUM
        p1 = Plots.heatmap(uv[:, :, num],
            color=:turbo,
            aspect_ratio=1,
            xlims=(0, Nx), ylims=(0, Ny),
            title=string("Velocity\n", "t = ", time[num]),
            xlabel="x",
            ylabel="y"
        )

        p2 = Plots.heatmap(Ω[:, :, num],
            color=:berlin,
            aspect_ratio=1,
            xlims=(0, Nx), ylims=(0, Ny),
            title=string("Vorticity\n", "t = ", time[num]),
            xlabel="x",
            ylabel="y"
        )

        p3 = Plots.heatmap(pressure[:, :, num],
            color=:vik,
            aspect_ratio=1,
            xlims=(0, Nx), ylims=(0, Ny),
            title=string("Pressure\n", "t = ", time[num]),
            xlabel="x",
            ylabel="y"
        )

        Plots.plot(p1, p2, p3, layout=(1, 3), size=(900, 650))
    end

    gif(anime, fps=15)
end
