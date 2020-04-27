#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <new>

/// Type of the kernel used by the SVM.
enum class KernelType {
  Linear,
  Polynomial,
  RBF,
  Sigmoid,
  /// Not implemented.
  Precomputed,
};

/// SVM type.
enum class SvmType {
  C_SVC,
  NU_SVC,
  ONE_CLASS,
  EPSILON_SVR,
  NU_SVR,
};

/// A two-class decision tree.
struct DecisionTree;

/// Libsvm uses a sparse representation of data,
/// where every entry in the training matrix
/// is characterised by a column index and a value.
/// Because this is a safe Rust-like object in itself,
/// it does not have a safe counterpart.
struct LibsvmNode {
  int32_t index;
  double value;
};

/// Libsvm structure representing training data.
struct LibsvmProblem {
  /// Number of rows in the training data.
  int32_t l;
  const double *y;
  /// Rows of the X matrix. Because row lenghts
  /// are not stored anywhere, and do not need
  /// to be equal, `libsvm` uses index = -1 as
  /// a sentinel value.
  const LibsvmNode *const *svm_node;
};

/// `libsvm` representation of training parameters.
struct LibsvmParameter {
  SvmType svm_type;
  KernelType kernel_type;
  int32_t degree;
  double gamma;
  double coef0;
  double cache_size;
  double eps;
  double C;
  int32_t nr_weight;
  const int32_t *weight_label;
  const double *weight;
  double nu;
  double p;
  int32_t shrinking;
  int32_t probability;
};

/// The model object returned from and accepted by
/// `libsvm` functions.
struct LibsvmModel {
  LibsvmParameter svm_parameter;
  int32_t nr_class;
  int32_t l;
  const LibsvmNode *const *SV;
  const double *const *sv_coef;
  const double *rho;
  const double *probA;
  const double *probB;
  const int32_t *sv_indices;
  const int32_t *label;
  const int32_t *nSV;
  int32_t free_sv;
};

extern "C" {

float decision_tree_predict(DecisionTree *decision_tree,
                            const char *test_x_url,
                            const char *test_y_url);

DecisionTree *decision_tree_train();

extern const char *svm_check_parameter(const LibsvmProblem *problem, const LibsvmParameter *param);

extern void svm_free_and_destroy_model(const LibsvmModel *const *svm_model);

extern double svm_predict_values(LibsvmModel *svm_model,
                                 const LibsvmNode *svm_nodes,
                                 const double *out);

extern const LibsvmModel *svm_train(const LibsvmProblem *prob, const LibsvmParameter *param);

} // extern "C"
