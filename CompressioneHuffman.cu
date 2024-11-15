#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <map>
#include <string.h>
#include <list>
#include <vector>
#include <bitset>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#define ASCIIDIM 2048


struct HuffmanNode
{
    HuffmanNode* left;
    HuffmanNode* right;
    char data;
    int freq;
    
    HuffmanNode(char d, int f) 
    :left(nullptr), right(nullptr), data(d), freq(f) {}
    HuffmanNode(int f, HuffmanNode* l, HuffmanNode* r)
    :left(l), right(r), data('\0'), freq(f) {}
};

void insertNode(std::list<HuffmanNode*>& nodeList, HuffmanNode* newNode) {
    auto it = nodeList.begin();
    while (it != nodeList.end() && (*it)->freq > newNode->freq) {
        ++it;
    }
    nodeList.insert(it, newNode);
}
HuffmanNode* buildHuffmanTree(std::list<HuffmanNode*>& nodeList) {

    while (nodeList.size() > 1) {

        HuffmanNode* left = nodeList.back();
        nodeList.pop_back(); 
        HuffmanNode* right = nodeList.back();
        nodeList.pop_back(); 
        HuffmanNode* parent = new HuffmanNode(left->freq + right->freq, left, right);
        insertNode(nodeList, parent);
    }
    return nodeList.front();
}
void generateHuffmanCodes(HuffmanNode* node, const std::string& currentCode, std::map<char, std::string>& huffmanCodes) {
    if (!node) return;

    if (!node->left && !node->right) {
        huffmanCodes[node->data] = currentCode;
    }

    generateHuffmanCodes(node->left, currentCode + "0", huffmanCodes);
    generateHuffmanCodes(node->right, currentCode + "1", huffmanCodes);
}
void printHuffmanCodes(const std::map<char, std::string>& huffmanCodes) {
    for (const auto& pair : huffmanCodes) {
        std::cout << "char: " << pair.first << ", cod: " << pair.second << "\n";
    }
}
void writeTree2(HuffmanNode* node, std::ofstream& file) {

    if(node==nullptr)
        return;
    file.write(reinterpret_cast<char*>(&node->data),sizeof(node->data));
    file.write(reinterpret_cast<char*>(&node->freq),sizeof(node->freq));
    bool hasChild = node->left!=nullptr || node->right!=nullptr;
    file.write(reinterpret_cast<char*>(&hasChild),sizeof(hasChild));
    if(node->left)
        writeTree2(node->left,file);
    if (node->right)
        writeTree2(node->right,file);
        
}

void writeTree(HuffmanNode* root, const std::string& filename, int nBits) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Errore nell'apertura del file di output." << std::endl;
        return;
    }
    int validBitsInLastByte = nBits % 8; 
    if (validBitsInLastByte == 0) {
        validBitsInLastByte = 8;  
    }

    file.write(reinterpret_cast<char*>(&validBitsInLastByte), sizeof(validBitsInLastByte));
    writeTree2(root, file);
    file.close();
}

__global__ void count_characters(const char* input, int* freq, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        atomicAdd(&freq[(char)input[idx]], 1);
    }
}

std::map<char, int> calculate_char_frequencies(const std::string& input, char** devCharOut) {
    int len = input.size();
    char* devChar;
    int* freq;
    int blockSize = 1024;
    int numBlocks = (len + blockSize - 1) / blockSize;

    cudaMalloc((void**)&devChar, len * sizeof(char));
    cudaMallocManaged(&freq, ASCIIDIM * sizeof(int));
    cudaMemset(freq, 0, ASCIIDIM * sizeof(int)); 
    cudaMemcpy(devChar, input.c_str(), len * sizeof(char), cudaMemcpyHostToDevice);

    count_characters<<<numBlocks, blockSize>>>(devChar, freq, len);
    cudaDeviceSynchronize();

    std::map<char, int> frequencyMap;
    for (int i = 0; i < ASCIIDIM; i++) {
        if (freq[i] > 0) {
            frequencyMap[(char)i] = freq[i];
        }
    }

    *devCharOut = devChar;  

    cudaFree(freq);  
    cudaDeviceSynchronize();
    return frequencyMap;
}

__global__ void encodeKernel(const char *text, int* codeLength, int* codeOffset, unsigned char* huffmanCode, unsigned char* bitStream, int textLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= textLength) { 
        return; 
    }
    int startOffset = codeOffset[idx];
    int codeLength1 = codeLength[idx];
    for (int i = 0; i < codeLength1; i++) {

        bitStream[startOffset + i] = huffmanCode[startOffset + i];
        
    }
}

std::vector<uint8_t> cudaEncode(char* devChar, std::string& text, const std::map<char, std::string>& huffmanCodes, int *nBits) {
    int textLength = text.length();
    std::vector<int> codeLengths(textLength);
    std::vector<int> codeOffsets(textLength);
    std::vector<unsigned char> huffmanCodes1;
    int bit = 0;

    for (int i = 0; i < textLength; i++) {
        try{
        const std::string& code = huffmanCodes.at(text[i]);
        codeLengths[i] = code.length();
        codeOffsets[i] = bit;
        for (char c : code) {
            if (c == '1')
                huffmanCodes1.push_back(1);
            else
                huffmanCodes1.push_back(0);
        }
        bit += code.size();
        }catch(std::out_of_range& e){
            printf("out of range\n");
        }
    }

    int* c_codeLengths;
    int* c_codeOffsets;
    unsigned char* c_huffmanCode;
    unsigned char* c_bitStream;

    cudaMalloc(&c_codeLengths, textLength * sizeof(int));
    cudaMalloc(&c_codeOffsets, textLength * sizeof(int));
    cudaMalloc(&c_huffmanCode, bit * sizeof(unsigned char));
    cudaMalloc(&c_bitStream, bit * sizeof(unsigned char));

    cudaMemcpy(c_codeLengths, codeLengths.data(), textLength * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_codeOffsets, codeOffsets.data(), textLength * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_huffmanCode, huffmanCodes1.data(), bit * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (textLength + blockSize - 1) / blockSize;
    
    encodeKernel<<<numBlocks, blockSize>>>(devChar, c_codeLengths, c_codeOffsets, c_huffmanCode, c_bitStream, textLength);

    std::vector<uint8_t> bitStream((bit + 7) / 8, 0);
    std::vector<unsigned char> tempBitStream(bit);
    cudaMemcpy(tempBitStream.data(), c_bitStream, bit * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < bit; i++) {
        if (tempBitStream[i] == 1) {
            bitStream[i / 8] |= (1 << (7 - (i % 8))); 
        }
    }

    cudaFree(c_codeLengths);
    cudaFree(c_codeOffsets);
    cudaFree(c_huffmanCode);
    cudaFree(c_bitStream);


    *nBits = bit;
    return bitStream;
}


int main(int argc, char* argv[]) {

    std::ifstream fileIn = std::ifstream(argv[1]);
    if (!fileIn.is_open()) {
        std::cerr << "Errore nell'apertura del file." << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << fileIn.rdbuf();  
    std::string text = buffer.str();  

    fileIn.close();

    char* devChar;//puntatore alla memoria del device
    std::map<char, int> freq = calculate_char_frequencies(text, &devChar);

    std::list<HuffmanNode*> nodeList;
    for (const auto& pair : freq) {
        HuffmanNode* newNode = new HuffmanNode(pair.first, pair.second);
        insertNode(nodeList, newNode);
    }

    HuffmanNode* root = buildHuffmanTree(nodeList);
    std::map<char, std::string> huffmanCodes;
    generateHuffmanCodes(root, "", huffmanCodes);

    int nBits = 0;
    std::vector<uint8_t> byteStream = cudaEncode(devChar, text, huffmanCodes, &nBits);

    std::ofstream fileOut("output.bin", std::ios::binary);
    if (!fileOut) {
        std::cout << "errore nell'apertura del file di output" << std::endl;
        return 1;
    }
    fileOut.write(reinterpret_cast<char*>(byteStream.data()), byteStream.size());
    fileOut.close();
    writeTree(root, "tree.bin", nBits);

    cudaFree(devChar);
    std::cout << "File di output compresso e albero Huffman scritti con successo." << std::endl;
    return 0;
}
