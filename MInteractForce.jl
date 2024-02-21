# 计算所有粒子的所有受力(流体作用力已给出)
using StatsBase, Combinatorics

# 排斥力函数
function InteractForce(xᵢ, yᵢ, xⱼ, yⱼ, A, B)
    #=
    (xᵢ, yᵢ)为受力粒子笛卡尔坐标；
    (xⱼ, yⱼ)为施力粒子笛卡尔坐标；
    A, B为排斥力控制参数;
    =#
    r = sqrt((xᵢ - xⱼ)^2 + (yᵢ - yⱼ)^2)
    if r <= B
        Force = A * (1 + cos(π / B * r))
        fx = (xᵢ - xⱼ) / r * Force
        fy = (yᵢ - yⱼ) / r * Force
    else
        fx = 0.0
        fy = 0.0
    end
    fVector = [fx, fy]

    return fVector
end

# 求解Force
function ForceSolver!(NumP::Int, k::Int, Force::Array{Float64,3}, XcYcK::Matrix, A, B, Nx, Ny, Mp, g)
    #=
    Force 已计入流体作用力
    XcYcK 为当前时间层k的全部粒子质心坐标
    =#
    list = collect(combinations(collect(1:NumP), 2))  # 粒子组合表
    nmax = size(list, 1)
    # 粒子间相互作用力...............................................................
    Fpp = [0.0, 0.0]
    for n in 1:nmax
        Fpp = InteractForce(XcYcK[1, list[n][1]], XcYcK[2, list[n][1]],
                            XcYcK[1, list[n][2]], XcYcK[2, list[n][2]],
                            A, B)
        Force[:, k+1, list[n][1]] = Force[:, k+1, list[n][1]] + Fpp
        Force[:, k+1, list[n][2]] = Force[:, k+1, list[n][2]] - Fpp
    end

    # 粒子与壁面作用力...............................................................
    for n in 1:NumP
        Force[:, k+1, n] = Force[:, k+1, n] +
                           InteractForce(XcYcK[1, n], 0, 0, 0, A, B) +
                           InteractForce(XcYcK[1, n], 0, Nx - 1, 0, A, B) +
                           InteractForce(0, XcYcK[2, n], 0, 0, A, B) +
                           InteractForce(0, XcYcK[2, n], 0, Ny - 1, A, B)
    end

    # 外力场........................................................................
    Force[:, k+1, :] = Force[:, k+1, :] .+ Mp * g
end