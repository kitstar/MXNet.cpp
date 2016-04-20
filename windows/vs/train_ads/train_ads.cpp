/*!
* Copyright (c) 2015 by Contributors
*/
# include "train_ads.h"

using namespace std;
using namespace mxnet::cpp;


size_t Mlp::Run(KVStore *kv, std::unique_ptr<dmlc::SeekStream> stream, size_t streamSize)
{

    /*define the symbolic net*/
    auto sym_x = Symbol::Variable("data");
    auto sym_label = Symbol::Variable("label");
    auto w1 = Symbol::Variable("w1");
    auto b1 = Symbol::Variable("b1");
    auto w2 = Symbol::Variable("w2");
    auto b2 = Symbol::Variable("b2");
    auto w3 = Symbol::Variable("w3");
    auto b3 = Symbol::Variable("b3");

    auto fc1 = FullyConnected("fc1", sym_x, w1, b1, 2048);
    auto act1 = Activation("act1", fc1, ActivationActType::relu);
    auto fc2 = FullyConnected("fc2", act1, w2, b2, 512);
    auto act2 = Activation("act2", fc2, ActivationActType::relu);
    auto fc3 = FullyConnected("fc3", act2, w3, b3, 1);
    auto mlp = LogisticRegressionOutput("softmax", fc3, sym_label);

    NDArray w1m(Shape(2048, 600), ctx_cpu),
        w2m(Shape(512, 2048), ctx_cpu),
        w3m(Shape(1, 512), ctx_cpu);
    NDArray::SampleGaussian(0, 1, &w1m);
    NDArray::SampleGaussian(0, 1, &w2m);
    NDArray::SampleGaussian(0, 1, &w3m);
    NDArray b1m(Shape(2048), ctx_cpu),
        b2m(Shape(512), ctx_cpu),
        b3m(Shape(1), ctx_cpu);
    NDArray::SampleGaussian(0, 1, &b1m);
    NDArray::SampleGaussian(0, 1, &b2m);
    NDArray::SampleGaussian(0, 1, &b3m);

    for (auto s : mlp.ListArguments()) {
        LG << s;
    }

    double samplesProcessed = 0;
    double sTime = get_time();
    int64_t pull_time_in_ms = 0;

    /*setup basic configs*/
    std::unique_ptr<Optimizer> opt(new Optimizer("ccsgd", learning_rate, weight_decay));
    (*opt).SetParam("momentum", 0.9)
        .SetParam("rescale_grad", 1.0 / (kv->GetNumWorkers() * batchSize));
    //.SetParam("clip_gradient", 10);
    if (kv->GetRank() == 0)
    {
        kv->SetOptimizer(std::move(opt));
    }
    kv->Barrier();
    
    bool init_kv = false;
    
    int my_rank = 0;
    int total_rank = 1;
    if (!is_local_data)
    {
        my_rank = kv->GetRank();
        total_rank = kv->GetNumWorkers();
    }

    if (kv->GetRole() == "worker")
    {
        for (int ITER = 0; ITER < maxEpoch; ++ITER) {
            NDArray testData, testLabel;
            int mb = 0;
            size_t totalSamples = 0;
            DataReader dataReader(stream.get(), streamSize,
                sampleSize, my_rank, total_rank, batchSize);
            while (!dataReader.Eof()) {
                //if (mb++ >= nMiniBatches) break;
                // read data in
                auto r = dataReader.ReadBatch();
                size_t nSamples = r.size() / sampleSize;
                totalSamples += nSamples;
                vector<float> data_vec, label_vec;
                samplesProcessed += nSamples;
                CHECK(!r.empty());
                for (int i = 0; i < nSamples; i++) {
                    float * rp = r.data() + sampleSize * i;
                    label_vec.push_back(*rp);
                    data_vec.insert(data_vec.end(), rp + 1, rp + sampleSize);
                }
                r.clear();
                r.shrink_to_fit();

                const float *dptr = data_vec.data();
                const float *lptr = label_vec.data();
                NDArray dataArray = NDArray(Shape(nSamples, sampleSize - 1),
                    ctx_cpu, false);
                NDArray labelArray =
                    NDArray(Shape(nSamples), ctx_cpu, false);
                dataArray.SyncCopyFromCPU(dptr, nSamples * (sampleSize - 1));
                labelArray.SyncCopyFromCPU(lptr, nSamples);
                args_map["data"] = dataArray;
                args_map["label"] = labelArray;
                args_map["w1"] = w1m;
                args_map["b1"] = b1m;
                args_map["w2"] = w2m;
                args_map["b2"] = b2m;
                args_map["w3"] = w3m;
                args_map["b3"] = b3m;
                Executor *exe = mlp.SimpleBind(ctx_dev, args_map);
                std::vector<int> indices(exe->arg_arrays.size());
                std::iota(indices.begin(), indices.end(), 0);
                if (!init_kv) {
                    kv->Init(indices, exe->arg_arrays);
                    kv->Pull(indices, &exe->arg_arrays);
                    init_kv = true;
                }
                exe->Forward(true);
                NDArray::WaitAll();
                LG << "Iter " << ITER
                    << ", accuracy: " << Auc(exe->outputs[0], labelArray)
                    << "\tsample/s: " << samplesProcessed / (get_time() - sTime)
                    << "\tProgress: [" << samplesProcessed * 100.0 / maxEpoch / dataReader.recordCount() << "%]";
                exe->Backward();
                kv->Push(indices, exe->grad_arrays);

                auto start_time = std::chrono::system_clock::now();

                kv->Pull(indices, &exe->arg_arrays);
                //exe->UpdateAll(&opt, learning_rate);
                NDArray::WaitAll();

                auto end_time = std::chrono::system_clock::now();
                auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                pull_time_in_ms += elapse_in_ms.count();

                delete exe;
            }
            LG << "Total samples: " << totalSamples << "\tPull Time: [" << pull_time_in_ms / 1000.0 << "s]";

            //LG << "Iter " << ITER
            //  << ", accuracy: " << ValAccuracy(mlp, testData, testLabel);
        }
    }
    else kv->Barrier();

    kv->Barrier();
    // if (kv->GetRank() == 0) output_model();

    return samplesProcessed;
}

void Mlp::output_model()
{
    std::ofstream modelout("mxnet_model.txt", std::ios::binary);

    for (auto it = args_map.begin(); it != args_map.end(); ++it)
    {
        int Bsize = 1;
        for (auto shape : it->second.GetShape())
        {
            Bsize *= shape;
        }

        modelout.write((char*)it->second.GetData(), sizeof(float) * Bsize);
    }
    modelout.close();
}

float Mlp::ValAccuracy(Symbol mlp,
    const NDArray& samples,
    const NDArray& labels)
{
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
}

float Mlp::Auc(const NDArray& result, const NDArray& labels) 
{
    result.WaitToRead();
    const mx_float *pResult = result.GetData();
    const mx_float *pLabel = labels.GetData();
    int nSamples = labels.GetShape()[0];
    size_t nCorrect = 0;
    for (int i = 0; i < nSamples; ++i) 
    {
        float label = pLabel[i];
        float p_label = pResult[i];        
        if (label == (p_label >= 0.5)) nCorrect++;        
    }
    return nCorrect * 1.0 / nSamples;
}



int main(int argc, const char *argv[])
{
    LG << "Usage: " << argv[0] << " training_data  machine_list  server_count_per_machine" << endl;
    CHECK_EQ(argc, 4);
    
    std::string args = "dist_async#";
    args += argv[2];
    args += '#';
    args += argv[3];    

    LG << "Train Ads running setting: " << args << "." << endl;
    KVStore *kv = new KVStore(args);
    kv->RunServer();

    using namespace dmlc::io;
    URI dataPath(argv[1]);

    if (dataPath.protocol == "hdfs://") init_env();

    auto hdfs = FileSystem::GetInstance(dataPath);    
    size_t size = hdfs->GetPathInfo(dataPath).size;
    std::unique_ptr<dmlc::SeekStream> stream(hdfs->OpenForRead(dataPath, false));

    Mlp mlp(dataPath.protocol.empty());
    auto start = std::chrono::steady_clock::now();
    auto sample_count = mlp.Run(kv, std::move(stream), size);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now() - start);
    LG << "Training Duration = " << duration.count() / 1000.0 << "s\tLocal machine speed: [" << sample_count * 1000.0 / duration.count() << "/s]\tTotal speed: [" << sample_count * 1000.0 * kv->GetNumWorkers() / duration.count() / 2 << "/s]";
    // delete kv;
}