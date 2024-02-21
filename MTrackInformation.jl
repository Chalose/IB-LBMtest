# 计算所有粒子的角速度、质心速度，质心坐标
function TrackSolver!(ωp::Matrix, up::Array{Float64, 3}, XcYc::Array{Float64, 3}, Force::Array{Float64, 3}, Torque::Matrix, k, δt, Mp, Ip)
    if k == 1
        ωp[:, 2] = ωp[:, 1] + δt / Ip * Torque[:, 1]
        up[:, 2, :] = up[:, 1, :] + δt / Mp * Force[:, 1, :]
        XcYc[:, 2, :] = XcYc[:, 1, :] + δt * up[:, 1, :]
    else
        ωp[:, k+1] = 4 / 3 * ωp[:, k] - 1 / 3 * ωp[:, k-1] + 2δt / (3Ip) * Torque[:, k+1]
        up[:, k+1, :] = 4 / 3 * up[:, k, :] - 1 / 3 * up[:, k-1, :] + 2δt / (3Mp) * Force[:, k+1, :]
        XcYc[:, k+1, :] = 4 / 3 * XcYc[:, k, :] - 1 / 3 * XcYc[:, k-1, :] + 2δt / 3 * up[:, k+1, :]
    end
end