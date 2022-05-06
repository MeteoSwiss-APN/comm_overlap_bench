#include "mpi.h"
#include <string>
#include <vector>
class MPIHelper {
    MPIHelper();
    static const MPIHelper& getInstance();

  public:
    static int rank();
    static int size();
    static void sendString(const std::string& message, int destination, int tag);
    static void sendString(const std::string& message, int tag);
    static std::string receiveString(int rank, int tag);
    static std::vector< std::string > send(const std::string& message, int tag);
    static void print(const std::string& prefix, const std::string& message, int tag);

  private:
    int rank_;
    int size_;
};
