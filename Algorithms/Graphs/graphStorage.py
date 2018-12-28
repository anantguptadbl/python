a=[[1,2,3,4,5,6,7,2],[7,3,4,8],[7,5,9]]

# Finding all edges
edges=set()
for x in a:
for y in range(len(x)-1):
  edges.add((x[y],x[y+1]))
  
# We will now find the neighbours
