#include <thrust/device_vector.h>   
#include <thrust/host_vector.h>     
#include <thrust/sort.h>            
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h> 
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>      
#include <thrust/transform.h>       
#include <thrust/for_each.h>        
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
std::map<char, int> calculate_char_frequencies(const std::string& input, char** devCharOut) {
    int len = input.size();
    thrust::device_vector<char> d_input(input.begin(), input.end());//copia valori su gpu
    *devCharOut = thrust::raw_pointer_cast(d_input.data());
    thrust::device_vector<int> d_freq(256, 0);
    thrust::for_each(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(len),[input_ptr = *devCharOut, freq_ptr = thrust::raw_pointer_cast(d_freq.data())] 
    __device__(int idx) {
            atomicAdd(&freq_ptr[(unsigned char)input_ptr[idx]], 1);
        });
    std::vector<int> h_freq(256);
    thrust::copy(d_freq.begin(), d_freq.end(), h_freq.begin());

    std::map<char, int> frequencyMap;//costruzione mappa
    for (int i = 0; i < 256; i++) {
        if (h_freq[i] > 0) {
            frequencyMap[(char)i] = h_freq[i];
        }
    }

    return frequencyMap;
}
std::vector<uint8_t> cudaEncode(char* devChar, const std::string& text, const std::map<char, std::string>& huffmanCodes, int* nBits) {
    int textLength = text.size();//preparazione vettori
    std::vector<int> codeLengths(textLength);
    std::vector<int> codeOffsets(textLength);
    std::vector<unsigned char> huffmanBits;
    int totalBits = 0;
    for (int i = 0; i < textLength; i++) {
        const std::string& code = huffmanCodes.at(text[i]);
        codeLengths[i] = code.length();
        codeOffsets[i] = totalBits;
        for (char c : code) {
            huffmanBits.push_back(c == '1' ? 1 : 0);
        }
        totalBits += code.length();
    }
    thrust::device_vector<int> d_codeLengths(codeLengths.begin(), codeLengths.end());//copia dati su gpu
    thrust::device_vector<int> d_codeOffsets(codeOffsets.begin(), codeOffsets.end());
    thrust::device_vector<unsigned char> d_huffmanBits(huffmanBits.begin(), huffmanBits.end());
    thrust::device_vector<unsigned char> d_bitStream(totalBits);

    thrust::for_each(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(textLength),
    [devChar, d_codeLengths_ptr = thrust::raw_pointer_cast(d_codeLengths.data()),
    d_codeOffsets_ptr = thrust::raw_pointer_cast(d_codeOffsets.data()),
    d_huffmanBits_ptr = thrust::raw_pointer_cast(d_huffmanBits.data()),
    d_bitStream_ptr = thrust::raw_pointer_cast(d_bitStream.data())] 
    __device__(int idx) {
            int startOffset = d_codeOffsets_ptr[idx];
            int codeLength = d_codeLengths_ptr[idx];
            for (int i = 0; i < codeLength; i++) {
                d_bitStream_ptr[startOffset + i] = d_huffmanBits_ptr[startOffset + i];
            }
        });
    std::vector<unsigned char> tempBitStream(totalBits);//copia e compattamento
    thrust::copy(d_bitStream.begin(), d_bitStream.end(), tempBitStream.begin());
    std::vector<uint8_t> bitStream((totalBits + 7) / 8, 0);
    for (size_t i = 0; i < totalBits; i++) {
        if (tempBitStream[i] == 1) {
            bitStream[i / 8] |= (1 << (7 - (i % 8)));
        }
    }

    *nBits = totalBits;
    return bitStream;
}
int main(int argc, char* argv[]) {

    std::ifstream fileIn = std::ifstream(argv[1]);
    if (!fileIn.is_open()) {
        std::cout << "Errore nell'apertura del file." << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << fileIn.rdbuf();  
    std::string text = buffer.str();  

    fileIn.close();

    char *devChar;//puntatore alla memoria del device
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
    std::vector<uint8_t> byteStream = cudaEncode(devChar,text, huffmanCodes, &nBits);

    std::ofstream fileOut("output.bin", std::ios::binary);
    if (!fileOut) {
        std::cout << "errore nell'apertura del file di output" << std::endl;
        return 1;
    }
    fileOut.write(reinterpret_cast<char*>(byteStream.data()), byteStream.size());
    fileOut.close();
    writeTree(root, "tree.bin", nBits);

    std::cout << "file di output compresso e albero Huffman scritti con successo." << std::endl;
    return 0;
}