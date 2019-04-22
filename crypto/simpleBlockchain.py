import time
import hashlib

class Block(object):
    def __init__(self,index,proof,previous_hash,data):
        self.index=index
        self.proof=proof
        self.previous_hash=previous_hash
        self.data=data
        self.timestamp=time.time()
        
    def get_hash(self):
        hashString="{}{}{}{}{}".format(self.index,self.proof,self.previous_hash,self.data,self.timestamp)
        return hashlib.sha256(hashString.encode()).hexdigest()

class BlockChain(object):
    def __init__(self):
        self.chain=[]
        self.current_block_data=[]
        self.create_genesis_block()
        
    def create_genesis_block(self):
        self.create_new_block(proof=0,previous_hash=0)
    
    def create_new_block(self,proof,previous_hash):
        # Creating a new block
        # INDEX          : This is taken from the length of the blocks in the chain
        # PROOF          : This is the proof that is added to the 
        # PREVIOUS_HASH  : The previous hash value is an integral part of the blockchain data
        # DATA           : This contains the list of all the individual data
        block=Block(
            index=len(self.chain),
            proof=proof,
            previous_hash=previous_hash,
            data=self.current_block_data
        )
        # Resetting the data
        self.current_block_data=[]
        # Appending the BLOCK to the chain
        self.chain.append(block)
        
    
    def create_new_datapoint(self,user,textData):
        self.current_block_data.append(
        {
            'user':user,
            'data':textData
        }
        )
        return self.get_last_block().index + 1
    
    def create_proof_of_work(self,previous_proof):
        # The proof that we have setup is that the proof should always be divisible by 7
        proof = previous_proof + 1  
        while (proof + previous_proof) % 7 != 0:  
            proof += 1  
        return proof
    
    def get_last_block(self):
        return self.chain[-1]
    
blockchain=BlockChain()
print(blockchain.chain)

last_block = blockchain.get_last_block()
last_proof = last_block.proof  
print(last_proof)
proof = blockchain.create_proof_of_work(last_proof)  
print(proof)

blockchain.create_new_datapoint(  
    user="user1",  
    textData="sampleText1"
)  

last_hash = last_block.get_hash()
block = blockchain.create_new_block(proof, last_hash)  
  
print(">>>>> After Mining...")  
print(blockchain.chain)
