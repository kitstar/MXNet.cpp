# include <mutex>
# include <string>
# include "seq2seq.h"
# include "util.h"
# include "data.h"
# include "chana_ps.h"

using namespace std;
using namespace mxnet::cpp;

int main(int argc, const char *argv[])
{
    LG << "Usage :" << argv[0] << "  machine_list  ps_per_machine" << endl;
    auto kv_args = Mlp::generate_kvstore_args(sync_mode_t::Async, argv[argc - 2], argv[argc - 1]);
    Seq2seq trainer;    

    return 0;
}