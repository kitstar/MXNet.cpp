/*!
*  Copyright (c) 2016 by Contributors
* \file base.h
* \brief base definitions for mxnetcpp
* \author Chuntao Hong, Zhang Chen
*/
# pragma once

# include <cstdlib>
# include "mxnet-cpp/c_api.h"

namespace mxnet {
namespace cpp {

typedef unsigned index_t;

enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
  * \brief perform an inplace write,
  * Target shares memory with one of input arguments.
  * This option only happen when
  */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};

}  // namespace cpp
}  // namespace mxnet