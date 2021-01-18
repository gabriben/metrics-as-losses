# adapted from https://github.com/ashrefm/multi-label-soft-f1.git
import pandas as pd

def removeRareLabels(x, sep, irrelevanceThreshold = None, nLabelsThreshold = None):

  # Get label frequencies in descending order
  label_freq = (x.apply(lambda s: str(s).split(sep)) # '|'
                .explode().value_counts().sort_values(ascending=False))

  if irrelevanceThreshold is not None:
      # Create a list of rare labels
      rare = list(label_freq[label_freq < irrelevanceThreshold].index)
      print("We will be ignoring these rare labels:", rare)
  elif nLabelsThreshold is not None:
      # Create a list of rare labels
      rare = list(label_freq[0 : (nLabelsThreshold + 1)].index)
      print("We will be ignoring these rare labels:", rare)    

      
  # Transform Genre into a list of labels and remove the rare ones
  x = (x.apply(lambda s: [l for l in str(s).split(sep) if l not in rare])) # '|'

  return(x)
