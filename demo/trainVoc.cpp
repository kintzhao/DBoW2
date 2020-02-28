#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <glog/logging.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace DBoW2;
using namespace std;

class VocTrain
{
public:
    VocTrain(const std::string&  img_dir, const std::string&  out_dir);

    int loadImgList();
    int loadFeatures();
    void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
    string creatVoc(const int branchs = 5, const int levels = 10, const std::string& voc_path ="voc", const WeightingType weight = TF_IDF, const ScoringType score = L1_NORM);
    void creatDatabase(const std::string& in_voc_path, const string &out_db_name);

private:
    vector<std::string > image_lists_;
    vector<vector<cv::Mat > > features_;
    std::string  img_dir_;
    std::string  out_dir_;
};

VocTrain::VocTrain(const std::string&  img_dir, const std::string&  out_dir):img_dir_(img_dir), out_dir_(out_dir)
{
    LOG(INFO)<<"Load image and detect features!!!";
    loadImgList();
    LOG(INFO)<<"Get image size: "<<image_lists_.size();
    loadFeatures();
    LOG(INFO)<<"Get features_ size: "<<features_.size();
}

int VocTrain::loadImgList()
{
    long number = 0;
    if(img_dir_.empty())
    {
        LOG(INFO)<<" dir_name is null ! ";
        return 0;
    }

    // check if dir_name is a valid directory
    struct stat s;
    lstat( img_dir_.c_str() , &s );
    if( ! S_ISDIR( s.st_mode ) )
    {
        LOG(INFO)<<"dir_name is not a valid directory !";
        return 0;
    }

    DIR * dir;                   // return value for opendir()
    dir = opendir( img_dir_.c_str());
    if( NULL == dir )
    {
        LOG(INFO)<<"Can not open dir "<<img_dir_;
        return 0;
    }
    LOG(INFO)<<"Successfully opened the dir !";

    struct dirent * filename;    // return value for readdir()
    while( ( filename = readdir(dir) ) != NULL )
    {
        // get rid of "." and ".."
        if( strcmp( filename->d_name , "." ) == 0 ||
            strcmp( filename->d_name , "..") == 0    )
            continue;

        std::string sFilename(filename->d_name);
        std::string suffixStr = sFilename.substr(sFilename.find_last_of('.') + 1);

        if (suffixStr.compare("png") == 0 || suffixStr.compare("jpeg") == 0 || suffixStr.compare("bmp") == 0 )
        {
            std::string img_path = img_dir_+"/"+filename->d_name;
            image_lists_.push_back(img_path);
            LOG(INFO)<<"map_info_name: "<<img_path<<std::endl;
            ++number;
        }

    }
    return image_lists_.size();

}

int VocTrain::loadFeatures()
{
    features_.clear();
    features_.reserve(image_lists_.size());

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    LOG(INFO) << "Extracting ORB features...";
    for(unsigned int i = 0; i < image_lists_.size(); ++i)
    {
      cv::Mat image = cv::imread(image_lists_[i], 0);
      cv::Mat mask;
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      orb->detectAndCompute(image, mask, keypoints, descriptors);

      features_.push_back(vector<cv::Mat >());
      changeStructure(descriptors, features_.back());
    }
    return  features_.size();
}

void VocTrain::changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);

    for(int i = 0; i < plain.rows; ++i)
    {
      out[i] = plain.row(i);
    }
}

std::string VocTrain::creatVoc(const int branchs, const int levels, const std::string& voc_path, const WeightingType weight, const ScoringType score)
{
    // branching factor and depth levels
    OrbVocabulary voc(branchs, levels, weight, score);

    LOG(INFO) << "Creating a small " << levels << "^" << levels << " vocabulary..." ;
    voc.create(features_);
    LOG(INFO) << " Finish voc!"  ;

    cout << "Vocabulary information: " << endl
    << voc << endl << endl;

    // lets do something with this vocabulary
    LOG(INFO) << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    unsigned int  test_num = 4 > features_.size()? features_.size() : 4;
    for(unsigned int  i = 0; i < test_num; i++)
    {
      voc.transform(features_[i], v1);
      for(unsigned int  j = 0; j < test_num; j++)
      {
        voc.transform(features_[j], v2);

        double score = voc.score(v1, v2);
        LOG(INFO)  << "TEST: Image " << i << " vs Image " << j << ": " << score << endl;
      }
    }

    // save the vocabulary to disk
    LOG(INFO) << "Saving vocabulary..." << endl;
    std::string save_path = out_dir_+"/"+voc_path+"_voc.yml.gz";
    voc.save(save_path);
    LOG(INFO) << "voc.save Done !";
    return save_path;
}

void VocTrain::creatDatabase(const std::string& in_voc_path, const std::string& out_db_name)
{
    LOG(INFO) << "Creating a small database..." << endl;

    // load the vocabulary from disk
    OrbVocabulary voc(in_voc_path);

    OrbDatabase db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(unsigned int i = 0; i < features_.size(); i++)
    {
      db.add(features_[i]);
    }

    LOG(INFO) << " db.add done!" << endl;
    LOG(INFO) << "Database information: " << endl << db << endl;

    // and query the database
    LOG(INFO) << "Querying the database: " << endl;

    QueryResults ret;
    unsigned int  test_num = 4 > features_.size()? features_.size() : 4;
    for(unsigned int i = 0; i < test_num; i++)
    {
      db.query(features_[i], ret, 4);

      // ret[0] is always the same image in this case, because we added it to the database. ret[1] is the second best match.
      LOG(INFO) << "Searching for Image " << i << ". " << ret << endl;
    }

    // we can save the database. The created file includes the vocabulary and the entries added
    std::string out_path = out_dir_+"/"+out_db_name+"_db.yml.gz";
    LOG(INFO) << "Saving database..." <<out_path<< endl;
    db.save(out_path);
    LOG(INFO) << "... done!" << endl;

    // once saved, we can load it again
    LOG(INFO) << "Retrieving database once again..." << endl;
    OrbDatabase db2(out_path);
    LOG(INFO) << "Construct new OrbDatabase" << db2 << endl;
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    //google::SetLogDestination(google::GLOG_INFO, "/tmp/vocTrain_");
    FLAGS_alsologtostderr =true;
    FLAGS_colorlogtostderr =true;

    VocTrain  vocTrain(argv[1], argv[2]);
    const std::string voc_path = vocTrain.creatVoc(std::atoi(argv[3]), std::atoi(argv[4]), argv[5]);

    vocTrain.creatDatabase(voc_path, argv[5]);
    return 0;
}
