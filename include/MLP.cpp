# include "MLP.h"

using namespace std;
using namespace mxnet::cpp;

/* static */ string Mlp::generate_kvstore_args(sync_mode_t mode, std::string machine_list, std::string ps_per_machine)
{
    string ret;
    switch (mode)
    {
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
