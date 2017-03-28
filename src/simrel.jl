# Simulation of Linear Model Data

# Starting Simrel Code
function simrel{T}(
    n::T         = 100,
    p::T         = 10,
    q::T         = 5,
    relpos::T    = [1, 2, 3],
    gamma::T     = 0.2,
    R2::T        = 0.9;
    ntest::T     = nothing,
    muY::T       = nothing,
    muX::T       = nothing,
    lambdaMin::T = 10e-4,
    sim::T       = nothing
)
    m = length(relpos) ## Number of relevant components
    if (q < m)
        error("Number of relevant predictors must at least equal to $(m).")
    end
    irrelPosition = setdiff(collect(1:p), relpos)
    extraDim = q - m
    predPosition = vcat(relpos, sample(irrelPosition, extraDim,
                                    replace = false, ordered = true))
    nu = lambdaMin * exp(-gamma)/(1 - lambdaMin)
    if (lambdaMin < 0 || lambdaMin >= 1)
        error("Parameter lambdaMin must be in between 0 and 1")
    end
    lambdas = (exp(-gamma * vec(1:p)) + nu) / (exp(-gamma) + nu)
    sigmaZ = diagm(lambdas)
    sigmaZinv = inv(sigmaZ)
    sigmaZy = zeros(p, 1)
    uniformSample = rand(Uniform(-1, 1), m)
    covarianceSample = sign(uniformSample) .* sqrt(R2 * abs(uniformSample) /
             sum(abs(uniformSample)) .* lambdas[relpos])
    sigmaZy[relpos] = covarianceSample
    sigmaY = 1
    sigma = vcat(hcat(sigmaY, sigmaZy'), hcat(sigmaZy, sigmaZ))

    # Rotation Matrix

    Q = randn(q, q) |> x -> x .- mean(x, 1) # Centering columns
    Rq = qr(Q)[:1]
    R = eye(p)
    R[predPosition, predPosition] = Rq
    if (q < (p - 1))
      Q = randn(p - q, p - q) |> x -> x .- mean(x, 1)
      Rnq = qr(Q)[:1]
      notPredPos = setdiff(1:p, predPosition)
      R[notPredPos, notPredPos] = Rnq
    end

    # Regression coefficients
    betaZ = sigmaZinv * sigmaZy
    betaX = R * betaZ
    beta0 = 0

    if !(muY == nothing)
      beta0 = beta0 + muY
    end
    if !(muX == nothing)
      beta0 = beta0 - transpose(betaX) * muX
    end

    R2 = transpose(sigmaZy) * betaZ
    minerror = sigmaY - R2

    if all(eig(sigma)[:1] .> 0)
      sigmaRot = chol(sigma)
      ucal = randn(n, p + 1)
      ucal_1 = ucal * sigmaRot
      Y = ucal_1[:, 1]
      if !(muY == nothing)
        Y = Y .+ muY
      end
      Z = ucal_1[:, 2:end]
      X = Z * transpose(R)
      if !(muX == nothing)
        X = X .+ muX
      end

      # Test Observations
      if !(ntest == nothing)
        utest = randn(ntest, p + 1)
        utest_1 = utest * sigmaRot
        testY = utest_1[:, 1]
        if !(muY == nothing)
          testY = testY .+ muY
          testZ = utest_1[:, 2:end]
          testX = testZ * transpose(R)
        end
        if !(muX == nothing)
          testX = testX .+ muX
        end
      else
        testX = nothing
        testY = nothing
      end
    else
      return error("Correlation matrix is not positive definite")
    end

    return @NT(
      beta0    = beta0,
      beta     = betaX,
      sigma    = sigma,
      R2       = R2,
      minerror = minerror,
      X        = X,
      Y        = Y,
      testX    = testX,
      testY    = testY
    )
end
