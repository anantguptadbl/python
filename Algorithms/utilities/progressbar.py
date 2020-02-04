from tqdm import tqdm
import time

# For loop
for i in tqdm(range(100)):
    time.sleep(0.01)
    
# List Comprehension
vals=[time.sleep(0.01) for x in tqdm(range(100))]
