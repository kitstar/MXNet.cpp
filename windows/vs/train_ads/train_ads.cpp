/*!
* Copyright (c) 2015 by Contributors
*/
# include "train_ads.h"
# include "MxNetCpp.h"
# include "util.h"
# include "data.h"

using namespace std;
using namespace mxnet::cpp;


TrainAds::TrainAds()
{
    batchSize = 300;
    sampleSize = 601;
    epochCount = 1;

    grad_req_type.resize(8);
    grad_req_type[0] = kNullOp;
    grad_req_type[1] = kWriteTo;
    grad_req_type[2] = kWriteTo;
    grad_req_type[3] = kWriteTo;
    grad_req_type[4] = kWriteTo;
    grad_req_type[5] = kWriteTo;
    grad_req_type[6] = kWriteTo;
    grad_req_type[7] = kNullOp;

    in_args.resize(8);
    arg_grad_store.resize(8);
}

void TrainAds::build_network()
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
    mlp = LogisticRegressionOutput("softmax", fc3, sym_label);
}


/* virtual */ size_t TrainAds::train(std::string file_name, std::string kvstore_args)
{
    size_t processed_sample_count = 0;

    // Init Paramter Server
    LG << "Train Ads running setting: " << kvstore_args << "." << endl;
    auto kv = new KVStore(kvstore_args);
    kv->RunServer();
    sync_mode_t mode;
    if (kvstore_args[5] == 'a')
    {
        // Parameter Server        
        mode = sync_mode_t::Async;
    }
    else
    {
        // Allreduce
        mode = sync_mode_t::Sync;
    }    

    build_network();    


    NDArray w1m(Shape(2048, 600), ctx_cpu, false),
        w2m(Shape(512, 2048), ctx_cpu, false),
        w3m(Shape(1, 512), ctx_cpu, false);

    NDArray w1m_g(Shape(2048, 600), ctx_cpu, false),
        w2m_g(Shape(512, 2048), ctx_cpu, false),
        w3m_g(Shape(1, 512), ctx_cpu, false);

    NDArray b1m(Shape(2048), ctx_cpu, false),
        b2m(Shape(512), ctx_cpu, false),
        b3m(Shape(1), ctx_cpu, false);

    NDArray b1m_g(Shape(2048), ctx_cpu, false),
        b2m_g(Shape(512), ctx_cpu, false),
        b3m_g(Shape(1), ctx_cpu, false);

    if (mode == sync_mode_t::Sync)
    {
        if (kv->GetRank() == 0)
        {
            NDArray::SampleGaussian(0, 1, &w1m);
            NDArray::SampleGaussian(0, 1, &w2m);
            NDArray::SampleGaussian(0, 1, &w3m);

            NDArray::SampleGaussian(0, 1, &b1m);
            NDArray::SampleGaussian(0, 1, &b2m);
            NDArray::SampleGaussian(0, 1, &b3m);
        }
        else
        {
            std::vector<mx_float> w1mdata(2048 * 600, 0);
            w1m.SyncCopyFromCPU(w1mdata);
            std::vector<mx_float> w2mdata(512 * 2048, 0);
            w2m.SyncCopyFromCPU(w2mdata);
            std::vector<mx_float> w3mdata(512, 0);
            w3m.SyncCopyFromCPU(w3mdata);

            std::vector<mx_float> b1mdata(2048, 0);
            b1m.SyncCopyFromCPU(b1mdata);
            std::vector<mx_float> b2mdata(512);
            b2m.SyncCopyFromCPU(b2mdata);
            std::vector<mx_float> b3mdata(1, 0);
            b3m.SyncCopyFromCPU(b3mdata);
        }
    }
    else
    {
        NDArray::SampleGaussian(0, 1, &w1m);
        NDArray::SampleGaussian(0, 1, &w2m);
        NDArray::SampleGaussian(0, 1, &w3m);

        NDArray::SampleGaussian(0, 1, &b1m);
        NDArray::SampleGaussian(0, 1, &b2m);
        NDArray::SampleGaussian(0, 1, &b3m);
    }

    for (auto s : mlp.ListArguments()) {
        LG << s;
    }
    
    auto start_time = std::chrono::system_clock::now();

    /*setup basic configs*/
    std::unique_ptr<Optimizer> opt(new Optimizer("ccsgd", learning_rate, weight_decay));
    (*opt).SetParam("momentum", 0.9)
        .SetParam("rescale_grad", 1.0 / (kv->GetNumWorkers() * batchSize));
    //.SetParam("clip_gradient", 10);
    
    if (mode == sync_mode_t::Async)
    {
        if (kv->GetRank() == 0) kv->SetOptimizer(std::move(opt));        
        kv->Barrier();
    }
            
    in_args.resize(8);
    arg_grad_store.resize(8);    

    in_args[1] = w1m;
    in_args[2] = b1m;
    in_args[3] = w2m;
    in_args[4] = b2m;
    in_args[5] = w3m;
    in_args[6] = b3m;

    arg_grad_store[0] = NDArray();
    arg_grad_store[1] = w1m_g;
    arg_grad_store[2] = b1m_g;
    arg_grad_store[3] = w2m_g;
    arg_grad_store[4] = b2m_g;
    arg_grad_store[5] = w3m_g;
    arg_grad_store[6] = b3m_g;
    arg_grad_store[7] = NDArray();

    bool init_kv = false;
    for (int ITER = 0; ITER < epochCount; ++ITER) 
    {                     
        DataReader *dataReader = get_file_reader(file_name, batchSize, sampleSize, kv->GetRank(), kv->GetNumWorkers());        

        while (!dataReader->Eof())
        {
            // read data in
            auto r = dataReader->ReadBatch();
            size_t nSamples = r.size() / sampleSize;
            vector<float> data_vec, label_vec;
            processed_sample_count += nSamples;
            CHECK(!r.empty());
            for (int i = 0; i < nSamples; i++) 
            {
                float * rp = r.data() + sampleSize * i;
                label_vec.push_back(*rp);
                data_vec.insert(data_vec.end(), rp + 1, rp + sampleSize);
            }
            r.clear();            

            const float *dptr = data_vec.data();
            const float *lptr = label_vec.data();
            NDArray dataArray = NDArray(Shape(nSamples, sampleSize - 1),
                ctx_cpu, false);
            NDArray labelArray =
                NDArray(Shape(nSamples), ctx_cpu, false);
            dataArray.SyncCopyFromCPU(dptr, nSamples * (sampleSize - 1));
            labelArray.SyncCopyFromCPU(lptr, nSamples);
            in_args[0] = dataArray;
            in_args[7] = labelArray;

            std::vector<NDArray> aux_states;

            Executor* exe = new Executor(mlp, ctx_dev, in_args, arg_grad_store,
                grad_req_type, aux_states);

            std::vector<int> indices(exe->arg_arrays.size());
            std::iota(indices.begin(), indices.end(), 0);
            if (!init_kv) 
            {
                if (mode == sync_mode_t::Sync)
                {
                    // Allreduce
                    std::vector<NDArray> parameters;
                    parameters.clear();
                    parameters.push_back(w1m);
                    parameters.push_back(w2m);
                    parameters.push_back(w3m);
                    parameters.push_back(b1m);
                    parameters.push_back(b2m);
                    parameters.push_back(b3m);
                    kv->AllReduce(&parameters);
                }
                else
                {
                    // Parameter Server
                    kv->Init(indices, exe->arg_arrays);

                    for (size_t i = 0; i < indices.size(); ++i)
                        if (grad_req_type[i] != kNullOp)
                        {
                            kv->Pull(indices[i], &exe->arg_arrays[i]);
                        }
                }
                
                init_kv = true;
            }

            exe->Forward(true);
            NDArray::WaitAll();

            auto cur_time = std::chrono::system_clock::now();
            auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time);

            LG << "Iter " << ITER
                << ", accuracy: " << Accuracy(exe->outputs[0], labelArray)
                << "\tsample/s: " << processed_sample_count * 1000.0 / elapse_in_ms.count()
                << "\tProgress: [" << processed_sample_count * 100.0 / epochCount / dataReader->recordCount() << "%]";
            
            exe->Backward();
            
            if (mode == sync_mode_t::Sync)
            {
                // Allreduce                
                kv->AllReduce(&exe->grad_arrays);
                for (size_t i = 0; i < indices.size(); i++)
                    if (grad_req_type[i] != kNullOp)
                    {
                        exe->grad_arrays[i].WaitToRead();
                        exe->arg_arrays[i].WaitToWrite();
                        opt->Update(indices[i], exe->arg_arrays[i], exe->grad_arrays[i]);
                    }
            }
            else
            {
                // Parameter Server
                for (size_t i = 0; i < indices.size(); ++i)
                    if (grad_req_type[i] != kNullOp)
                    {
                        kv->Push(indices[i], exe->grad_arrays[i]);
                        kv->Pull(indices[i], &exe->arg_arrays[i]);
                    }

                NDArray::WaitAll();
            }

            delete exe;
        }

        delete dataReader;
    }

    kv->Barrier();
    
    auto end_time = std::chrono::system_clock::now();
    auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);    
    LG << "Training Duration = " << elapse_in_ms.count() / 1000.0 << 
        "s\tLocal machine speed: [" << processed_sample_count * 1000.0 / elapse_in_ms.count() << 
        "/s]\tTotal speed: [" << processed_sample_count * 1000.0 * kv->GetNumWorkers() / elapse_in_ms.count() << "/s]";

    if (kv->GetRank() == 0)
    {
        if (mode == sync_mode_t::Async && kv->GetNumWorkers() > 0)
        {
            for (size_t i = 0; i < in_args.size(); ++i)
                if (grad_req_type[i] != kNullOp)
                {
                    kv->Pull(i, &in_args[i]);
                }

            NDArray::WaitAll();
        }

        output_model(output_file);
    }
        
    kv->Barrier();
    
    delete kv;    

    return processed_sample_count;
}


/* virtual */ size_t TrainAds::predict(string file_name, string model_name, string kvstore_args)
{    
    size_t processed_sample_count = 0;
    
    init_env(file_name);
    
    
    auto kv = new KVStore(kvstore_args);
        
    build_network();
    
    NDArray w1m(Shape(2048, 600), ctx_cpu, false),
        w2m(Shape(512, 2048), ctx_cpu, false),
        w3m(Shape(1, 512), ctx_cpu, false);

    NDArray w1m_g(Shape(2048, 600), ctx_cpu, false),
        w2m_g(Shape(512, 2048), ctx_cpu, false),
        w3m_g(Shape(1, 512), ctx_cpu, false);

    NDArray b1m(Shape(2048), ctx_cpu, false),
        b2m(Shape(512), ctx_cpu, false),
        b3m(Shape(1), ctx_cpu, false);

    NDArray b1m_g(Shape(2048), ctx_cpu, false),
        b2m_g(Shape(512), ctx_cpu, false),
        b3m_g(Shape(1), ctx_cpu, false);
    
    in_args[1] = w1m;
    in_args[2] = b1m;
    in_args[3] = w2m;
    in_args[4] = b2m;
    in_args[5] = w3m;
    in_args[6] = b3m;
    
    arg_grad_store[0] = NDArray();
    arg_grad_store[1] = w1m_g;
    arg_grad_store[2] = b1m_g;
    arg_grad_store[3] = w2m_g;
    arg_grad_store[4] = b2m_g;
    arg_grad_store[5] = w3m_g;
    arg_grad_store[6] = b3m_g;
    arg_grad_store[7] = NDArray();

    load_model(model_name);        
    
    auto start_time = std::chrono::system_clock::now();
    
    batchSize = 499 * 1024 / sizeof(mx_float);        
    DataReader * dataReader = get_file_reader(file_name, batchSize, sampleSize, kv->GetRank(), kv->GetNumWorkers());

    string temp_out_file = output_file + ".part" + to_string(kv->GetRank());    
    auto output_stream = dmlc::Stream::Create(temp_out_file.c_str(), "w", true);

    while (!dataReader->Eof())
    {        
        // read data in
        auto r = dataReader->ReadBatch();
        size_t nSamples = r.size() / sampleSize;
        vector<float> data_vec, label_vec;
        processed_sample_count += nSamples;
        CHECK(!r.empty());
        for (int i = 0; i < nSamples; i++) 
        {
            float * rp = r.data() + sampleSize * i;
            label_vec.push_back(*rp);
            data_vec.insert(data_vec.end(), rp + 1, rp + sampleSize);
        }
        r.clear();        

        const float *dptr = data_vec.data();
        const float *lptr = label_vec.data();
        NDArray dataArray = NDArray(Shape(nSamples, sampleSize - 1),
            ctx_cpu, false);
        NDArray labelArray =
            NDArray(Shape(nSamples), ctx_cpu, false);
        dataArray.SyncCopyFromCPU(dptr, nSamples * (sampleSize - 1));
        labelArray.SyncCopyFromCPU(lptr, nSamples);
        in_args[0] = dataArray;
        in_args[7] = labelArray;

        std::vector<NDArray> aux_states;

        Executor* exe = new Executor(mlp, ctx_dev, in_args, arg_grad_store,
            grad_req_type, aux_states);

        std::vector<int> indices(exe->arg_arrays.size());
        std::iota(indices.begin(), indices.end(), 0);        

        exe->Forward(false);
        NDArray::WaitAll();

        auto cur_time = std::chrono::system_clock::now();
        auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time);
        
        LG  << ", accuracy: " << Accuracy(exe->outputs[0], labelArray)
            << "\tsample/s: " << processed_sample_count * 1000.0 / elapse_in_ms.count()
            << "\tProgress: [" << processed_sample_count * 100.0 / epochCount / dataReader->recordCount() << "%]";

        int sample_count = labelArray.GetShape()[0];
        
        output_stream->Write(reinterpret_cast<const void *>(exe->outputs[0].GetData()), sample_count * sizeof(mx_float));
                        
        delete exe;
    }
    delete output_stream;
    
    auto end_time = std::chrono::system_clock::now();
    auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    LG << "Predicting Duration = " << elapse_in_ms.count() / 1000.0 <<
        "s\tLocal machine speed: [" << processed_sample_count * 1000.0 / elapse_in_ms.count() <<
        "/s]\tTotal speed: [" << processed_sample_count * 1000.0 * kv->GetNumWorkers() / elapse_in_ms.count() << "/s]";

    kv->Barrier();

    if (output_file.substr(0, 7) == "hdfs://")
    {
        if (kv->GetRank() == 0)
        {
            merge_files(output_file, kv->GetNumWorkers());
        }
        kv->Barrier();
    }
        
    delete kv;
    kv = nullptr;

    return processed_sample_count;
}

/* virtual */ void TrainAds::load_model(string model_name)
{
    std::ifstream fin(model_name, std::ios::binary);

    for (auto it = 0; it < in_args.size(); ++it)
        if (grad_req_type[it] != kNullOp)
        {
            int Bsize = 1;
            for (auto shape : in_args[it].GetShape())
            {
                Bsize *= shape;
            }

            fin.read((char*)in_args[it].GetData(), sizeof(float) * Bsize);
        }

    fin.close();
}


/* virtual */ void TrainAds::output_model(std::string model_name)
{
    dmlc::Stream *stream = nullptr;
    size_t s;
    get_file_stream(model_name, stream, &s, "w");    

    for (auto it = 0; it < in_args.size(); ++it)
        if (grad_req_type[it] != kNullOp)
        {
            int Bsize = 1;
            for (auto shape : in_args[it].GetShape())
            {
                Bsize *= shape;
            }

            stream->Write((char*)in_args[it].GetData(), sizeof(float) * Bsize);
        }
    
    delete stream;
}


int main(int argc, const char *argv[])
{    
    LG << "Usage:";
    LG << "\tFor training: " << argv[0] << " train_data  output_model  machine_list  server_count_per_machine" << endl;
    LG << "\tFor predicting: " << argv[0] << " predict_data  input_model  predict_result machine_list  server_count_per_machine" << endl;
    CHECK_GE(argc, 4);
    
# if !defined(NDEBUG)
    getchar();
# endif
    
    auto kv_args = Mlp::generate_kvstore_args(sync_mode_t::Async, argv[argc - 2], argv[argc - 1]);
    TrainAds trainer;
    if (argc == 5)
    {
        // Training
        trainer.output_file = argv[2];
        trainer.train(argv[1], kv_args);
    }
    else
    {
        // Predicting
        trainer.output_file = argv[3];
        trainer.predict(argv[1], argv[2], kv_args);
    }

    return 0;
}