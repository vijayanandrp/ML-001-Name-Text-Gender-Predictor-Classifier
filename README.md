# #ML-001-Gender-Name-Predictor-Classifier

My first project I have learned in Machine Learning.

Forked Stephen Holiday's (Special Thanks & Kudos to him) project and migrated to Python3.5 for learning purpose.

**Sample Male Names Dataset - (name, count(name treated as a male), count(name treated as a female))**
```plain
[('SEYMOUR', 13030, 10), ('ARMANNI', 35, 11), ('TAJAI', 155, 147), ('JOSHUS', 171, 0), ('KUNTE', 26, 0), ('OTIES', 17, 0), ('RAYBORN', 48, 0), ('UBAID', 13, 0), ('GOTTLOB', 5, 0), ('LEVIT', 10, 0)]
```
**Sample Female Names Dataset - (name, count(treated as a male), count(treated as a female))**
```plain
[('TALESA', 0, 37), ('BRIGIDA', 0, 654), ('JOHNESE', 0, 41), ('DAMAYAH', 0, 18), ('JOSHELIN', 0, 14), ('GAOLEE', 0, 5), ('CANDIDA', 0, 3551), ('TAYHA', 0, 15), ('STEEVIE', 0, 5), ('CHRISOULA', 0, 97)]
```
**Probability Distribution - Calculation for Each Name**
```plain
male_probability = total_male_count / (total_male_count + total_female_count)

[Some Rules/Condition below this line. Seriously I don't know why we do this. 
But all that I know is no system can hold the value 1.0 in probability]

if male_probability == 1.0:
	male_probability = 0.99
elif male_probability == 0.0:
	male_probability = 0.01

female_probability = 1.0 - male_probability
```

**Sample Feature Set**
```plain
({
	'male_prob': 0.9992331288343558, 
	'last_two': 'UR', 
	'last_is_vowel': False, 
	'last_letter': 'R', 
	'female_prob': 0.0007668711656442229, 
	'last_three': 'OUR'}, 
'M')

({
	'male_prob': 0.99, 
	'last_two': 'WA', 
	'last_is_vowel': True, 
	'last_letter': 'A', 
	'female_prob': 0.010000000000000009, 
	'last_three': 'EWA'}, 
'M')

```


**Output - Preview**
```plain
name.pickle exists, loading data
name.pickle loaded
32031 males names loaded, 56347 female loaded

Accuracy: 0.9692803801765105

Most Informative Features
	 last_three = INA 
	 last_two = CK 
	 last_three = CIA 
	 last_three = ENA 
	 last_three = NNA 
	 last_three = IRA 
	 last_three = ICK 
	 last_three = SIA 
	 last_three = TTA 
	 last_three = ICA 
   
 <<<  Testing Module   >>> 
Enter "q" or "quit" to end testing module
Enter name to classify: vijay
vijay is classified as M
Enter name to classify: anand
anand is classified as M
Enter name to classify: anna
anna is classified as F
Enter name to classify: uma
uma is classified as F
Enter name to classify: quit
End
```

Please feel free to SHARE YOUR THOUGHTS !!

# genderPredictor #
GenderPredictor is a wrapper around [NLTK](http://www.nltk.org/)'s [Naive Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier for predicting the gender given a name.

This problem is common when dealing with incomplete contact information for users.

Currently it appears to be about 82% accurate on American names but this is just the framework.
The name files are from the [US Social Security Administration](http://www.ssa.gov/oact/babynames/limits.html) and are likely in the public domain. The processed files are distributed under the same rules as the original data (which is likely public domain...).

The code is under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) license.

Comments and suggestions are welcome at [stephen.holiday@gmail.com](mailto:stephen.holiday@gmail.com),

Stephen Holiday
[stephenholiday.com](http://stephenholiday.com)
