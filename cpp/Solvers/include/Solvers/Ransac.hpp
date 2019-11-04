#pragma once

#include <random>


namespace pcv
{

template<class DataType, class ModelType>
ModelType
Ransac(
    const std::vector<DataType> &data,
    const std::function<ModelType(std::vector<DataType>)> &modelingFunction,
    std::size_t dataPointsRequiredByModel,
    const std::function<double(ModelType, DataType)> &costFunction,
    double errorThreshold,
    std::size_t iterations)
{
    if (data.size() < dataPointsRequiredByModel)
    {
        throw std::runtime_error("There must be at least the minimum number of "
                                 "data points required by modeling function.");
    }

    std::size_t bestInlierCount = 0;

    // The default value of bestCost shall never be used. It should always
    // get overwritten otherwise discarded.
    double bestCost = 0;

    ModelType bestModel{};

    // Always run at least one iteration of RANSAC.
    for (std::size_t iteration = 0; iteration < iterations+1; ++iteration)
    {
        std::vector<DataType> tmpInliers;
        for (std::size_t minDataIter = 0; minDataIter < 4; ++minDataIter)
        {
            std::uniform_int_distribution<std::size_t> indexRandomizer(0, data.size());
            std::random_device randomDevice;
            tmpInliers.push_back(data[indexRandomizer(randomDevice)]);
        }

        auto tmpModel = modelingFunction(tmpInliers);
        std::size_t tmpInlierCount = 0;

        const auto tmpTotalCost = std::accumulate(
            std::cbegin(data),
            std::cend(data),
            0.0,
            [tmpModel, errorThreshold, &tmpInlierCount, &costFunction](double accumulator, DataType dataIter)
            {
                const auto &error = costFunction(tmpModel, dataIter);

                // Do not count this point's error toward the total cost because
                // it is an outlier.
                if (error > errorThreshold)
                {
                    return accumulator;
                }

                ++tmpInlierCount;
                return accumulator + error;
            });


        // If the current model has most inliers insofar, or if the the number
        // of inliers matches the most inliers insofar and the cost has been
        // reduced, then update the best model with the current model.
        if (tmpInlierCount > bestInlierCount
           || ((tmpInlierCount == bestInlierCount) && tmpTotalCost < bestCost))
        {
            bestCost = tmpTotalCost;
            bestInlierCount = tmpInlierCount;
            bestModel = tmpModel;
        }
    }
    return bestModel;
}

} // end namespace pcv
