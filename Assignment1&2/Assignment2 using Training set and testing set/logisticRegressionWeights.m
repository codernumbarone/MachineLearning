function [w] = logisticRegressionWeights( XTrain, yTrain, w0, maxIter, learningRate)
 
    [nSamples, nFeature] = size(XTrain);
    w = w0;
    precost = 0;
    for j = 1:maxIter

        temp = zeros(nFeature + 1,1);
        for k = 1:nSamples
            temp = temp + (sigmoid([1.0 XTrain(k,:)] * w) - yTrain(k)) * [1.0 XTrain(k,:)]';
        end
        w = w - learningRate * temp;
        cost = CostFunc(XTrain, yTrain, w);
        if j~=0 && abs(cost - precost) / cost <= 0.0001
            break;
        end

        precost = cost;

    end


end
