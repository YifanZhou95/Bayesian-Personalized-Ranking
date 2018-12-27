import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
for l in readGz("train.json.gz"):
  user,business = l['userID'],l['businessID']
  allRatings.append(l['rating'])
  userRatings[user].append(l['rating'])

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if u in userAverage:
    predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
  else:
    predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')

predictions.close()

### Would-visit baseline: just rank which businesses are popular and which are not, and return '1' if a business is among the top-ranked

businessCount = defaultdict(int)
totalPurchases = 0

for l in readGz("train.json.gz"):
  user,business = l['userID'],l['businessID']
  businessCount[business] += 1
  totalPurchases += 1

mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPurchases/2: break

predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if i in return1:
    predictions.write(u + '-' + i + ",1\n")
  else:
    predictions.write(u + '-' + i + ",0\n")

predictions.close()

### Category prediction baseline: Just consider some of the most common words from each category

catDict = {
  "American Restaurant": 0,
  "Bar": 1,
  "Asian Restaurant": 2,
  "European Restaurant": 3,
  "Italian Restaurant": 4,
  "Fast Food Restaurant": 5,
  "Mexican Restaurant": 6,
  "Seafood Restaurant": 7,
  "Coffee Shop": 8,
  "Sandwich Shop": 9
}

predictions = open("predictions_Category.txt", 'w')
predictions.write("userID-reviewHash,category\n")
for l in readGz("test_Category.json.gz"):
  cat = catDict['American Restaurant'] # If there's no evidence, just choose the most common category in the dataset
  words = l['reviewText'].lower()
  if 'america' in words:
    cat = catDict['American Restaurant']
  if 'bar' in words or 'beer' in words:
    cat = catDict['Bar']
  if 'asia' in words:
    cat = catDict['Asian Restaurant']
  if 'europe' in words:
    cat = catDict['European Restaurant']
  if 'italian' in words:
    cat = catDict['Italian Restaurant']
  if 'fast' in words:
    cat = catDict['Fast Food Restaurant']
  if 'mexic' in words:
    cat = catDict['Mexican Restaurant']
  if 'coffee' in words:
    cat = catDict['Coffee Shop']
  if 'sandwich' in words:
    cat = catDict['Sandwich Shop']
  predictions.write(l['userID'] + '-' + l['reviewHash'] + "," + str(cat) + "\n")

predictions.close()
