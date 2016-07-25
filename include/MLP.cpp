# include "MLP.h"
# include <unordered_set>
# include "util.h"

using namespace std;
using namespace mxnet::cpp;


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


/* static */ size_t Mlp::BuildVocabulary(const std::string &vocab_file, const std::string &corpus_file, std::unordered_map<std::string, uint32_t> &vocab_map)
{
    string s;
    size_t idx = 0;
    
    ifstream fin(vocab_file.c_str());    
    if (fin.good())
    {
        
        while (getline(fin, s))
        {
            vocab_map[s] = idx++;
        }
        fin.close();
        return idx;
    }
    else
    {
        fin.close();
        fin.clear();
        unordered_map<string, bool> word_set;

        fin.open(corpus_file.c_str());
        while (fin >> s)
        {
            word_set[s] = true;
        }
        fin.close();

        ofstream fout(vocab_file.c_str());
        for (auto &e : word_set)
        {
            fout << e.first << endl;
        }
        fout.close();
    }
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
    GetFileStream(model_name, stream, &s, "r");

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
    GetFileStream(model_name, stream, &s, "w");

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
    // For binary output
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