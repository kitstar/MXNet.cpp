# include "lstm.h"
# include "mxnet-cpp/kvstore.h"
# include "util.h"

using namespace std;
using namespace mxnet::cpp;

int main(int argc, const char *argv[])
{
    LG << "Usage:";
    LG << "\tFor distributed training: " << argv[0] << " machine_list  server_count_per_machine" << endl;

    auto chana_config = getenv("CHANA_CONFIG_FILE");
    if (chana_config != nullptr)
    {
        std::string config(chana_config);
        config = "config=" + config;
        const char *temp_argv[] = { argv[0], config.c_str() };
        chana_initialize(2, temp_argv);
    }
    else
    {
        std::string config("config=config.ini");
        const char *temp_argv[] = { argv[0], config.c_str() };
        chana_initialize(2, temp_argv);
    }

    LSTM ptb;
    KVStore *kv_store = nullptr;
    if (argc > 2) kv_store = Mlp::InitializeKvstore(ptb.running_mode_, argv[1], argv[2]);
    else kv_store = Mlp::InitializeKvstore(ptb.running_mode_, "", "");

    ptb.Run(kv_store);

    return 0;
}