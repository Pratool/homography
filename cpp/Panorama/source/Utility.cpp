#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "Panorama/Utility.hpp"

cv::Mat distortSourceToMatchTarget(
    const cv::Mat &sourceImage, const cv::Mat &targetImage)
{
    if (! sourceImage.data || ! targetImage.data)
    {
        throw std::runtime_error("Null image data.");
    }

    std::vector<cv::KeyPoint> sourceKeyPoints;
    cv::Mat sourceDescriptors;

    std::vector<cv::KeyPoint> targetKeyPoints;
    cv::Mat targetDescriptors;

    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(targetImage,
                            cv::noArray(),
                            targetKeyPoints,
                            targetDescriptors);
    akaze->detectAndCompute(sourceImage,
                            cv::noArray(),
                            sourceKeyPoints,
                            sourceDescriptors);




    cv::Mat sourceRawKeyPointsImage;
    cv::drawKeypoints(sourceImage,
                      sourceKeyPoints,
                      sourceRawKeyPointsImage);
    cv::imwrite("sourceRawKeyPoints.png", sourceRawKeyPointsImage);
    std::clog << "wrote sourceRawKeyPoints.png" << std::endl;
    cv::Mat targetRawKeyPointsImage;
    cv::drawKeypoints(targetImage,
                      targetKeyPoints,
                      targetRawKeyPointsImage);
    cv::imwrite("targetRawKeyPoints.png", targetRawKeyPointsImage);
    std::clog << "wrote targetRawKeyPoints.png" << std::endl;


    cv::BFMatcher bruteForceMatcher{cv::NORM_HAMMING};

    std::vector< std::vector<cv::DMatch> > nearestNeighborMatches;
    bruteForceMatcher.knnMatch(
            targetDescriptors,
            sourceDescriptors,
            nearestNeighborMatches,
            2); // Get only the 2 best nearest-neighbor matches. Useful for Lowe's ratio.
    std::clog << "finished running matcher" << std::endl;


    // Create intermediate vector storing only the best matches.
    std::vector<cv::DMatch> rawMatches;
    for (const auto &matchesVec : nearestNeighborMatches)
    {
        rawMatches.push_back(matchesVec[0]);
    }
    // Display the best nearest-neighbor matches for visual inspection.
    cv::Mat rawMatchesImage;
    cv::drawMatches(targetImage,
                    targetKeyPoints,
                    sourceImage,
                    sourceKeyPoints,
                    rawMatches,
                    rawMatchesImage);
    cv::imwrite("rawMatches.png", rawMatchesImage);
    std::clog << "wrote rawMatches.png" << std::endl;


    // Get all nearest neighbor matches where the best and second-best matches
    // are not too far apart.
    std::vector<cv::DMatch> matches;
    for (const auto &matchVec : nearestNeighborMatches)
    {
        const auto &best = matchVec[0];
        constexpr const double nearestNeighborRatio = 0.8;

        if (matchVec[0].distance < nearestNeighborRatio * matchVec[1].distance)
        {
            matches.push_back(best);
        }
    }
    std::clog << "filtered matches" << std::endl;


    // Display the filtered best nearest-neighbor matches for visual inspection.
    cv::Mat filteredMatchesImage;
    cv::drawMatches(targetImage,
                    targetKeyPoints,
                    sourceImage,
                    sourceKeyPoints,
                    matches,
                    filteredMatchesImage);
    cv::imwrite("filteredMatches.png", filteredMatchesImage);
    std::clog << "wrote filteredMatches.png" << std::endl;


    // Get the pixel coordinates of each of the matches.
    std::vector<cv::Point2f> sourceRansacInliers;
    std::vector<cv::Point2f> targetRansacInliers;
    for (const auto &match : matches)
    {
        targetRansacInliers.push_back(targetKeyPoints[match.queryIdx].pt);
        sourceRansacInliers.push_back(sourceKeyPoints[match.trainIdx].pt);
    }

    std::clog << "inliers to total: "
              << matches.size() << "/" << sourceKeyPoints.size()
              << " = " << matches.size()/double(sourceKeyPoints.size())
              << std::endl;

    cv::Mat homography = cv::findHomography(targetRansacInliers,
                                            sourceRansacInliers,
                                            cv::RANSAC,
                                            3);
    std::clog << "found homography matrix" << std::endl;


    cv::Mat sourceWarped;
    cv::warpPerspective(sourceImage, sourceWarped, homography, sourceImage.size());
    cv::imwrite("warpedSource.png", sourceWarped);

    std::clog << "wrote warpedSource.png" << std::endl;

    return sourceWarped;
}
