#include "MPIHelper.h"
#include <iostream>

MPIHelper::MPIHelper() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

const MPIHelper& MPIHelper::getInstance() {
    static MPIHelper instance;
    return instance;
}

int MPIHelper::rank() {
    return getInstance().rank_;
}

int MPIHelper::size() {
    return getInstance().size_;
}

void MPIHelper::sendString(const std::string& message, int tag) {
    sendString(message, 0, tag);
}

void MPIHelper::sendString(const std::string &message, int destination, int tag) {
    MPI_Send(message.c_str(), message.length(), MPI_CHAR, destination, tag, MPI_COMM_WORLD);
}

std::string MPIHelper::receiveString(int source, int tag) {
    MPI_Status status;
    MPI_Probe(source, tag, MPI_COMM_WORLD, &status);

    int count;
    MPI_Get_count(&status, MPI_CHAR, &count);

    // Fill buffer
    std::vector<char> buffer(count);
    MPI_Recv(&(buffer[0]), count, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);

    std::string msg(&(buffer[0]), count);
    return msg;
}

std::vector<std::string> MPIHelper::send(const std::string& message, int tag) {

    if (rank() == 0) {
        std::vector<std::string> messages;
        messages.reserve(size());

        // Handle rank 0
        messages.push_back(message);
        for (size_t i=1; i<size(); ++i) {
            messages.push_back(receiveString(i, tag));
        }
        return messages;
    }
    
    sendString(message, tag);
    std::vector<std::string> empty;
    return empty;
}

void MPIHelper::print(const std::string &prefix, const std::string &message, int tag) {
    auto messages = send(message, tag);
    if (messages.size() > 0) {
        std::cout << prefix;
        for (const auto& m: messages) {
            std::cout << m;
        }
        std::cout << std::endl;
    }
}
