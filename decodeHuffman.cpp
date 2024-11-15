#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <map>
#include <string.h>
#include <list>
#include <vector>
#include <bitset>
#include <sstream>
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
HuffmanNode * loadTree2(std::fstream & file){
    char data;
    int freq;
    bool figlio;
    file.read(reinterpret_cast<char*>(&data),sizeof(data));
    file.read(reinterpret_cast<char*>(&freq),sizeof(freq));

    file.read(reinterpret_cast<char*>(&figlio),sizeof(figlio));
    HuffmanNode * newnode = new HuffmanNode(data,freq);

    if(figlio){
        newnode->left = loadTree2(file);
        newnode->right = loadTree2(file);
    }


    return newnode;
}
HuffmanNode * loadTree(const std::string & filename, int * validBitsInLastByte){
    std::fstream file(filename);
    if(!file.is_open()){
        std::cout<<"Errore nell'apertura del file dell'albero"<<std::endl;
        return nullptr;
    }
    int validBitsInLastByte1;
    file.read(reinterpret_cast<char*>(&validBitsInLastByte1), sizeof(validBitsInLastByte1));
    //std::cout<<validBitsInLastByte1<<std::endl;
    *validBitsInLastByte = validBitsInLastByte1;
    HuffmanNode * root = loadTree2(file);
    file.close();
    return root;
}
std::string decodeByteStream(const std::vector<uint8_t>& byteStream, HuffmanNode* root, int totalBits, int validBitsInLastByte) {
    std::string decodedText;
    HuffmanNode* currentNode = root;

    int bitCount = 0;
    int byteCount = byteStream.size();

    for (int byteIndex = 0; byteIndex < byteCount; ++byteIndex) {
        uint8_t byte = byteStream[byteIndex];

        int bitsToProcess = (byteIndex == byteCount - 1) ? validBitsInLastByte : 8;

        for (int i = 7; i >= 8 - bitsToProcess && bitCount < totalBits; --i) {  
            bool bit = (byte >> i) & 1;  

            if (bit == 0) {
                currentNode = currentNode->left;
            } else {
                currentNode = currentNode->right;
            }

            if (!currentNode->left && !currentNode->right) {
                decodedText += currentNode->data;
                currentNode = root;
            }

            bitCount++;
        }
    }
    return decodedText;
}


int main(){
    int validBitsInLastByte;
    HuffmanNode * root = loadTree("tree.bin",&validBitsInLastByte);

    std::ifstream fileIn2 ("output.bin",std::ios::binary);
    if(!fileIn2){
        std::cout<<"errore nell'apertura del file di input 2";
        return 0;
    }
    fileIn2.seekg(0, std::ios::end);
    std::streamsize fileSize = fileIn2.tellg();
    fileIn2.seekg(0, std::ios::beg);

    std::vector<uint8_t> byteStream2(fileSize);
    if(fileIn2.read(reinterpret_cast<char*>(byteStream2.data()), fileSize)){}else{
        std::cout<<"errore lettura del bytestream";
    }
    //decodifica
    std::string decodedText = decodeByteStream(byteStream2, root, byteStream2.size()*8, validBitsInLastByte);
    std::cout << "Decodificato: " << decodedText << "\n";
    return 0;
}