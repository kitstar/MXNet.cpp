# include "MLP.h"
# include "io/filesys.h"
# include "MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;
using namespace dmlc;


void get_file_stream(const std::string &file_name, dmlc::Stream *&stream, size_t *file_size, const char *op)
{
    if (strchr(op, 'r') != nullptr)
    {        
        io::URI file_uri(file_name.c_str());
        auto fs = io::FileSystem::GetInstance(file_uri);
        *file_size = fs->GetPathInfo(file_uri).size;
        stream = fs->OpenForRead(file_uri, false);
    }
    else
    {
        file_size = 0;
        stream = Stream::Create(file_name.c_str(), op, true);
    }
}

DataReader * get_file_reader(const std::string &file_name, int buffer_sample_count, int sample_size, int my_rank, int total_rank)
{
    if (file_name.substr(0, 7) != "hdfs://")
    {
        my_rank = 0;
        total_rank = 1;
    }
    
    Stream *stream = nullptr;
    size_t file_size = 0;
    get_file_stream(file_name, stream, &file_size, "r");
    auto ret = new DataReader(dynamic_cast<SeekStream *>(stream), file_size, sample_size, my_rank, total_rank, buffer_sample_count);
    return ret;
}

void merge_files(std::string final_name, int count)
{
    Stream *final_stream = nullptr;
    size_t file_size;
    size_t readed_size = 0;
    const size_t buf_size = 64 * 1024 * 1024;
    char *buf = (char *)malloc(buf_size + 64);

    get_file_stream(final_name, final_stream, &file_size, "w");

    for (int i = 0; i < count; ++i)
    {
        string cur_file = final_name + ".part" + to_string(i);
        Stream *cur_stream;
        get_file_stream(cur_file, cur_stream, &file_size, "r");

        while ((readed_size = cur_stream->Read(buf, buf_size)) > 0)
        {
            final_stream->Write(buf, readed_size);
        }
        delete cur_stream;        
    }
    
    free(buf);
    delete(final_stream);
}


/* static */ string Mlp::generate_kvstore_args(sync_mode_t mode, std::string machine_list, std::string ps_per_machine)
{
    string ret;
    switch (mode)
    {
    case (sync_mode_t::Local) :
        return "local";        
    
    case (sync_mode_t::Sync) :
        ret += "dist_sync#";
        break;

    case (sync_mode_t::Async) :
        ret += "dist_async#";
        break;
    
    default:
        LG << "Invalid parameter server sync mode!";
        return ret;
    }

    ret += machine_list + "#" + ps_per_machine;

    return ret;
}

double Mlp::Accuracy(const NDArray& result, const NDArray& labels)
{
    result.WaitToRead();
    auto pResult = result.GetData();
    auto pLabel = labels.GetData();
    int sample_count = labels.GetShape()[0];
    size_t nCorrect = 0;
    for (int i = 0; i < sample_count; ++i)
    {
        auto label = pLabel[i];
        auto p_label = pResult[i];
        if (label == (p_label >= 0.5)) nCorrect++;
    }
    return nCorrect * 1.0 / sample_count;
}

double Mlp::ValAccuracy(Symbol mlp,
    const NDArray& samples,
    const NDArray& labels)
{
    /*
    size_t nSamples = samples.GetShape()[0];
    size_t nCorrect = 0;
    size_t startIndex = 0;
    args_map["data"] = samples;
    args_map["label"] = labels;

    Executor *exe = mlp.SimpleBind(ctx_dev, args_map);
    exe->Forward(false);
    const auto &out = exe->outputs;
    NDArray result = out[0].Copy(ctx_cpu);
    result.WaitToRead();
    const mx_float *pResult = result.GetData();
    const mx_float *pLabel = labels.GetData();
    for (int i = 0; i < nSamples; ++i) {
        float label = pLabel[i];
        int cat_num = result.GetShape()[1];
        float p_label = 0, max_p = pResult[i * cat_num];
        for (int j = 0; j < cat_num; ++j) {
            float p = pResult[i * cat_num + j];
            if (max_p < p) {
                p_label = j;
                max_p = p;
            }
        }
        if (label == p_label) nCorrect++;
    }
    delete exe;

    return nCorrect * 1.0 / nSamples;
    */
    return 0;
}
