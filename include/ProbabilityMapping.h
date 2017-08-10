/*
 * =====================================================================================
 *
 *       Filename:  ProbabilityMapping.cc
 *
 *    Description:
 *
 *        Version:  0.1
 *        Created:  01/21/2016 10:39:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Josh Tang, Rebecca Frederick
 *
 *        version: 1.0
 *        created: 8/9/2016
 *        Log: fix a lot of bug, Almost rewrite the code
 *
 *        author: He Yijia
 *
 * =====================================================================================
 */

#ifndef PROBABILITYMAPPING_H
#define PROBABILITYMAPPING_H

#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <numeric>
//#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <mutex>

#define covisN 7
#define covisn 3
#define sigmaI 20
#define lambdaG 30//30
#define lambdaL 60//75
#define lambdaTheta 45  // 35
#define lambdaN 3//8
#define histo_length 30
#define th_high 100
#define th_low 50
#define THETA 0.23 //0.43
#define NNRATIO 0.6

#define NULL_DEPTH 999

namespace ORB_SLAM2 {
class KeyFrame;
class Map;
}


namespace cv {
class Mat;
}

class ProbabilityMapping {
public:

    struct depthHo {
        float depth;
        float sigma;
                Eigen::Vector3f Pw; // point pose in world frame
                bool supported;
                depthHo():depth(0.0),sigma(0.0),Pw(0.0, 0.0, 0.0),supported(false){}
        };
    struct three_pair{
        int a;
        float b;
        float c;
        three_pair():a(0),b(0),c(0){}
    };

struct seed{
float a;
float b;
float sigma2;
float mu;
float d_range;
seed():a(20),b(10),sigma2(0.),mu(0.),d_range(0.){}
};
        ProbabilityMapping(ORB_SLAM2::Map *pMap);

        void Run();
        // add some const depth point into key frame
        void TestSemiDenseViewer();
     //   float bilinear1(const cv::Mat& img, const float& y, const float& x);
        /* * \brief void SemiDenseLoop(ORB_SLAM2::KeyFrame kf, depthHo**, std::vector<depthHo>*): return results of epipolar search (depth hypotheses) */
        void SemiDenseLoop();
        /* * \brief void stereo_search_constraints(): return min, max inverse depth */
        void StereoSearchConstraints(ORB_SLAM2::KeyFrame* kf, float* min_invdepth, float* max_invdepth);
    /* * \brief void epipolar_search(): return distribution of inverse depths/sigmas for each pixel */
        void EpipolarSearch(ORB_SLAM2::KeyFrame *kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel, float min_invdepth, float max_invdepth, depthHo *dh, cv::Mat F12, float& pjju,float& pjjv,float &best_u, float &best_v,float th_pi,float&a,float&b,float&c);
        void GetSearchRange(float& umin, float& umax, int px, int py, float mininvd, float maxinvd, ORB_SLAM2::KeyFrame* kf, ORB_SLAM2::KeyFrame* kf2);
        /* * \brief void inverse_depth_hypothesis_fusion(const vector<depthHo> H, depthHo* dist):
     * *         get the parameters of depth hypothesis distrubution from list of depth hypotheses */
        void InverseDepthHypothesisFusion(const std::vector<depthHo>& h, depthHo &dist);
    /* * \brief void intraKeyFrameDepthChecking(std::vector<std::vector<depthHo> > h, int imrows, int imcols): intra-keyframe depth-checking, smoothing, and growing. */
        void IntraKeyFrameDepthChecking(std::vector<std::vector<depthHo> >& ho, int imrows, int imcols);
        //  vector is low.  use depth_map and detph_sigma (cv::Mat)  to speed
        void IntraKeyFrameDepthChecking(cv::Mat& depth_map, cv::Mat& depth_sigma, const cv::Mat gradimg);
        /* * \brief void interKeyFrameDepthChecking(ORB_SLAM2::KeyFrame* currentKF, std::vector<std::vector<depthHo> > h, int imrows, int imcols):
         * *         inter-keyframe depth-checking, smoothing, and growing. */
        void InterKeyFrameDepthChecking(const cv::Mat& im, ORB_SLAM2::KeyFrame* currentKF, std::vector<std::vector<depthHo> >& h);
        void InterKeyFrameDepthChecking(ORB_SLAM2::KeyFrame* currentKf,int&biaozhi);
        void IntraKeyFrameDepthChecking1(cv::Mat& depth_map, cv::Mat& depth_sigma,const cv::Mat gradimg);
   void updateseed(std::vector<depthHo>&depth_ho,seed &seed1,depthHo &depth_temp);
   void EpipolarSearch_after(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
       float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float&pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&a,float&b,float&c);
   void EpipolarSearch_h(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
       float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float&pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&a,float&b,float&c);
   void EpipolarSearch_after1(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
       float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float&pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&a,float&b,float&c);
   void EpipolarSearch_after07(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
       float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float&pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&a,float&b,float&c);
   void getWarpMatrixAffine(ORB_SLAM2::KeyFrame*kf1,ORB_SLAM2::KeyFrame*kf2,int u,int v,float depth,int halfpatch_size,cv::Mat&A_kf2_kf1);
   float rubang(float a,float b);
   cv::Mat kaifang(cv::Mat &a);
   float fanwei(float a);
   cv::Mat getnormal(ORB_SLAM2::KeyFrame *kf,float x,float y);
   void Findpipei1(ORB_SLAM2::KeyFrame *kf1,std::vector<ORB_SLAM2::KeyFrame*> closestMatches,int x, int y,float pixel,float min_invdepth,
                                      float max_invdepth,int num_step,float &best_invdepth,float &best_sagma);
   void findpixel1(ORB_SLAM2::KeyFrame *kf1,ORB_SLAM2::KeyFrame *kf2,float pixel,float inv_depth,int x,int y,float &pixel_huidu_gradient);
        void RequestFinish()
        {
            //unique_lock<mutex> lock(mMutexFinish);
            mbFinishRequested = true;
        }

        bool CheckFinish()
        {
            //unique_lock<mutex> lock(mMutexFinish);
            return mbFinishRequested;
        }

private:
        bool mbFinishRequested;
        ORB_SLAM2::Map* mpMap;
        float bilinear1(const cv::Mat& img, const float& y, const float& x);
        void GetTR(ORB_SLAM2::KeyFrame* kf, cv::Mat* t, cv::Mat* r);
        void GetXp(const cv::Mat& K, int x, int y, cv::Mat* Xp);
        void GetParameterization(const cv::Mat& F12, const int x, const int y, float &a, float &b, float &c);
        void ComputeInvDepthHypothesis(ORB_SLAM2::KeyFrame* kf, ORB_SLAM2::KeyFrame *kf2, float umin,float umax,float ustar, float ustar_var, float a, float b, float c, depthHo *dh, int x, int y);
        void GetGradientMagAndOri(const cv::Mat& image, cv::Mat* gradx, cv::Mat* grady, cv::Mat* mag, cv::Mat* ori);
        void GetInPlaneRotation(ORB_SLAM2::KeyFrame* k1, ORB_SLAM2::KeyFrame* k2, float* th);
        void PixelNeighborSupport(std::vector<std::vector<depthHo> > H, int x, int y, std::vector<depthHo>& support);
        void PixelNeighborNeighborSupport(std::vector<std::vector<depthHo> > H, int px, int py, std::vector<std::vector<depthHo> >& support);
        void GetIntensityGradient_D(const cv::Mat& ImGrad, float a, float b, float c, int px, float* q);
        void GetPixelDepth(float uj, int px, int py, ORB_SLAM2::KeyFrame* kf, ORB_SLAM2::KeyFrame *kf2, float &p);
        void GetPixelDepth(float uj, float vj, int px, int py, ORB_SLAM2::KeyFrame* kf, ORB_SLAM2::KeyFrame* kf2, float &p, depthHo *dh);
        bool ChiTest(const depthHo& ha, const depthHo& hb, float* chi_val);
        bool ChiTest(const float& a, const float& b, const float sigma_a, float sigma_b);
        //void GetFusion(const std::vector<depthHo>& best_compatible_ho, depthHo* hypothesis, float* min_sigma);
        void GetFusion(const std::vector<std::pair <float,float> > supported, float& depth, float& sigma);
        void GetFusion(const std::vector<depthHo>& best_compatible_ho, depthHo& hypothesis, float* min_sigma);
        void Equation14(depthHo& dHjn, float& depthp, cv::Mat& xp, cv::Mat& rji, cv::Mat& tji, float* res);
        cv::Mat ComputeFundamental(ORB_SLAM2::KeyFrame *&pKF1, ORB_SLAM2::KeyFrame *&pKF2);
        cv::Mat GetSkewSymmetricMatrix(const cv::Mat &v);
        void updataDepthFilter(const std::vector<depthHo> &depth_ho,depthHo &depth_temp);
        void findbestvecdepth(const std::vector<depthHo>& h, std::vector<depthHo>& dist);
        float geman_mcclure(float x);
protected:
            std::mutex mMutexSemiDense;
};

#endif
