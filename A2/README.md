# CP640 - Assignment 2
### Ryan Kazmerik (175826410)

*This project requires Python 2.7 to run, as well as
the breast-cancer-wisconsin.data text file located
in the /data directory of the project root*

## Question 2
Implement k-nearest neighbours using any programming language of your choice. Use the UCI Breast Cancer Wisconsin (Original) as the testbed 1. Classify each input x according to the most frequent class amongst its k nearest neighbours. Break ties at random. Test the algorithms by 10-fold cross validation. What to hand in:
* Your code for k-nearest neighbours and cross validation.
* Find the best k by 10-fold cross validation. Draw a graph that shows the
accuracy as k increases from 1 to 30.

## Data Quality Plan
Dataset: UCI Breast Cancer Wisconsin (Original)
Filename: breast-cancer-wisconsin.data
Rows: 699

<br/>
<table>
  <tr>
    <th>Feature</th>
    <th>Data Quality Issue</th>
    <th>Handling Strategy</th>
  </tr>
  <tr>
    <td>Single Epithelial Cell Size</td>
    <td>Missing values (2%)</td>
    <td>Complete case analysis</td>
  </tr>
</table>
<br/>

## Data Pre-processing
* Complete Case Analysis - Because the amount of missing data was only 2% for this feature, complete case analysis was used to remove those rows from the data set, resulting in 683 rows.

* Normalization - all numerical values for this data set were between 1 and 10, therefore no normalization was used as the data is already respectively scaled.

* Feature Removal - the PatientID column was removed from the dataset to prevent it being used in the knn distance calculations.