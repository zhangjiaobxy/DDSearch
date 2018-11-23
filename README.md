DDSearch 1.0 - Linux version
------------------------

DDSearch 1.0 is implemented in Python3.

This package includes DDSearch 1.0 Python scripts and related files for the following paper:

DDSearch: a deep learning based global network similarity search algorithm for dynamic networks
(Jiao Zhang, Sam Kwong, Zhaolei Zhang, Reemma Muthal Puredath, and Ka-Chun Wong)


CONTAINS:
------------------------

* src : a folder contains DDSearch 1.0 Python scripts and related executable binary. It has following files :

	* DDSearch.py : DDSearch 1.0 Python code

	* dynamagna : DynaMAGNA++ executable binary from paper (Vijayan, Vipin, Dominic Critchlow, and T. Milenković. "Alignment of dynamic networks." Bioinformatics 33.14 (2017): i180-i189.)
	
* data : this folder stores all the data related to DDSearch 1.0. It has following files :

	* rawData/db/ : it stores the input database networks of DDSearch 1.0, which exists in the initial stage

	* rawData/query/ : it stores the input query set networks of DDSearch 1.0, which exists in the initial stage

	* randomWalks : it stores all random walks for each dynamic network, which is created in the program execution stage

	* doc2vec : it stores all document embedding files, which is created in the program execution stage
	
	* NA : it stores label files of DynaMAGNA++, which is created in the program execution stage

	* feaLabel : it stores the csv file with feature and label information, which is created in the program execution stage

	* output : it stores top k percent similar networks and terminal output information, which is created in the program execution stage

* requirements.txt : pre-installed packages for DDSearch

* runExample.sh : an example of the command line to run DDSearch

* LICENSE : MIT License

* README.md : this file


PREREQUISITE
------------------------

DDSearch 1.0 was tested using Python 3.4.3 version on Ubuntu 14.04 LTS. Python packages in the requirements.txt should be installed before running DDSearch:

	$ pip install -r requirements.txt


HOW TO USE
------------------------

* Examples:
	
	$ cd /DDSearch/src/
	
	$ chmod +x *
	
	$ python DDSearch.py --topk 10 --label DS3

* Parameters:

	* [d] : integer number, it is the feature dimension in the document embedding

	* [l] : integer number, it is the walk length per vertex in second order random walk

	* [r] : integer number, it is the random walks per vertex in second order random walk

	* [c] : integer number, it is the window size in the document embedding

	* [p] : float number, it is the BFS controller in second order random walk

	* [q] : float number, it is the DFS controller in second order random walk

	* [beta] : float number, it is the ratio parameter to control weight of Time and BLAST score

	* [topk] : integer number, it is the topk percent similar networks in the output

	* [label] : boolen type, it could be DS3, ICS, or EC


FILE NAMING RULE
------------------------

* Input network file is named in the txt file format. File name is interger number. File name in the database should not have any interaction with the query set file name.

* For example, if there are 100 networks in the database and 10 query networks in the query set. Therefore, file names in the database should start from 1.txt to 100.txt. File names in the query set should start from 101.txt to 110.txt.


FILE FORMAT OF DDSearch
------------------------

* Input network is in the txt file format, which has the following format: 

 Each line corresponds to an interaction and contains: the name of the first vertex, the name of the second vertex, the time of the first vertex, the time of the second vertex, the BLAST score. The vertices can be presented by integer or string.

 Here is an example for network "1.txt" :

	vertex1	vertex2	timeVertex1|timeVertex2|BLASTScore

	3	5	0.1|10.34|31.2

	6	4	0.89|14.2|1

	8	9	1.48|10.35|1

	11	1	5.12|14.59|1

	10	2	6.84|11.76|1

	7	4	9.96|13.15|1

* Output file is in the csv file format, which has the following format: 

 The output file returns top K similar networks against each query network. Each line contains the name of the query network, the name of similar netowrk in the database, label for regression, and prediction of the similarity score (separated by a comma).

 Here is an example for the output file "dynamagna_fea_DS3_top10%.csv" :

	q_name,db_name,dynamagna_DS3_label,predictions

	101,70,0.048835,0.30376145

	101,14,0.018706,0.28369987

	101,46,0.0,0.27936572

	101,62,0.038182,0.25859562


FUNDING SUPPORT
------------------------
* We would like to thank Amazon Web Service (AWS) for providing cloud credits for the software development.


------------------------
Jiao Zhang

jiaozhang9-c@my.cityu.edu.hk

November 23, 2018

