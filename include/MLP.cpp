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

DataReader * get_file_reader(const std::string &file_name, int buffer_sample_count, int sample_size, int my_rank, int total_rank, size_t skip_offset)
{
    if (file_name.substr(0, 7) != "hdfs://")
    {
        my_rank = 0;
        total_rank = 1;
    }
    
    Stream *stream = nullptr;
    size_t file_size = 0;
    get_file_stream(file_name, stream, &file_size, "r");
    auto ret = new DataReader(dynamic_cast<SeekStream *>(stream), file_size, sample_size, my_rank, total_rank, buffer_sample_count, skip_offset);
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

/* static */ KVStore * Mlp::InitializeKvstore(sync_mode_t mode, std::string machine_list, std::string ps_per_machine)
{
    string ret;
    KVStore *kv = nullptr;
    switch (mode)
    {
    case (sync_mode_t::Local) :
        ret = "local";
        kv = new KVStore(ret);
        return kv;

    case (sync_mode_t::Sync) :
        ret = "dist_sync#";
        ret += machine_list + "#" + ps_per_machine;
        kv = new KVStore(ret);        
        return kv;        

    case (sync_mode_t::Async) :
        ret = "dist_async#";
        ret += machine_list + "#" + ps_per_machine;
        kv = new KVStore(ret);
        kv->RunServer();
        return kv;

    default:
        LG << "Invalid parameter server sync mode!";        
    }
    return nullptr;
}

/* static */ mx_float Mlp::GetSoftmaxResult(const mx_float *prediction, int cat_num)
{
    mx_float ret = 0;
    mx_float max_p = prediction[0];        
    for (int i = 1; i < cat_num; ++i)    
        if (max_p < prediction[i])
        {
            ret = i;
            max_p = prediction[i];
        }
    
    return ret;
}

Mlp::Mlp() : ctx_cpu(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)),
    ctx_dev(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)),
    running_mode_(sync_mode_t::Local), running_op_(RunningMethods::kInvalid)
{
    auto mode = chana_config_get_value_string(mxnet_section.c_str(), "running_mode", "local", "");
    if (strcmp(mode, "async") == 0) running_mode_ = sync_mode_t::Async;
    else if (strcmp(mode, "sync") == 0) running_mode_ = sync_mode_t::Sync;

    auto op = chana_config_get_value_string(mxnet_section.c_str(), "operation", "", "");
    if (strcmp(op, "train") == 0) running_op_ = RunningMethods::kTrain;
    else if (strcmp(op, "predict") == 0) running_op_ = RunningMethods::kPredict;
    else if (strcmp(op, "validate") == 0) running_op_ = RunningMethods::kValidate;
}


/* virtual */ bool Mlp::LoadModel(const string &model_name, vector<NDArray> &parameters)
{
    LG << "Load model from [" << model_name << "]";
    
    dmlc::Stream *stream = nullptr;
    size_t s;
    get_file_stream(model_name, stream, &s, "r");
    
    mx_float *temp_store = nullptr;
    bool ret = true;

    for (size_t idx = 0; idx < parameters.size(); ++idx)
    {
        size_t matrix_size = 1;
        for (auto shape : parameters[idx].GetShape())
        {
            matrix_size *= shape;
        }        

        temp_store = reinterpret_cast<mx_float *>(malloc(matrix_size *  sizeof(mx_float)));
        auto readed_size = stream->Read(temp_store, matrix_size *  sizeof(mx_float));
        if (readed_size != matrix_size *  sizeof(mx_float))
        {
            ret = false;
        }
        
        parameters[idx].SyncCopyFromCPU(temp_store, matrix_size);
        parameters[idx].WaitToRead();
        free(temp_store);
    }

    delete stream;
    return ret;
}


/* virtual */ bool Mlp::SaveModel(const string &model_name, vector<NDArray> &parameters)
{
    LG << "Save model at " << model_name << endl;
    
    size_t s;
    dmlc::Stream *stream = nullptr;    
    get_file_stream(model_name, stream, &s, "w");        

    for (size_t idx = 0; idx < parameters.size(); ++idx)
    {
        size_t matrix_size = 1;
        for (auto shape : parameters[idx].GetShape())
        {
            matrix_size *= shape;
        }
        stream->Write(parameters[idx].GetData(), matrix_size * sizeof(mx_float));
    }
    
    delete stream;
    
    return true;
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