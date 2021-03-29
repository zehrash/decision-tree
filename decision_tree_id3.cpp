#include <math.h>
#include<vector>
#include <fstream>
#include <ostream>
#include <iostream>
#include <sstream>
#include <map>

using namespace std;

const int columnsForTrain = 6;
const int rowsForTrain = 891;
const int columnsForTest = 6;
const int rowsForTest = 418;

map < int, vector<int> > parsedData;
int selected[100000];
vector <int> cardinality;
vector <int> outcomeInput; // 0 for no ,1 for yes - contains the result of all m data points

struct Node {
	bool answer;
	int attribute;
	Node* next[10000];
};

Node* createNode(){
	Node* node = new Node;
	node->attribute = -1;
	node->answer = -1;
	return node;
}

class DecisionTreeImp {
	/*Calculates the amount of uncertainty in dataset*/
public:
	double calcEntropy(int* x, int* y) {
		if (*x == 0 || *y == 0)
			return 0;
		return -(((double)*x / (*x + *y)) * (log((double)*x / *x + *y) / log(2)) + ((double)*y / (*x + *y)) * (log((double)*y / *x + *y) / log(2)));
	}
	/*Calculates information gain*/
	double calcInfoGain(int attributeNumber, vector<int> data) {

		int positiveCount = 0;//Number of positive examples
		for (int i = 0; i < outcomeInput.size(); i++) {
			if (outcomeInput[i] == 1) {
				positiveCount++;
			}
		}//calculating positive outcome for survivours
		int negativeCount = outcomeInput.size() - positiveCount;
		double initial = calcEntropy(&positiveCount, &negativeCount);//initial entropy of original set

		double final = 0.0;

		for (int j = 0; j < cardinality[attributeNumber]; j++) {
			int numOfPositivesInAttr = 0;// Number of positive examples with attribute value j
			int numOfNegativesInAttr = 0;//Number of negative examples with attribute value j
			for (int i = 0; i < data.size(); i++)
				if (parsedData[attributeNumber][data[i]] == j && outcomeInput[data[i]] == 1) {//checking if data from dataset has the same unique value 
																							//and if the outcome value(survivours) is positive
					numOfPositivesInAttr++;
				}
				else if (parsedData[attributeNumber][data[i]] == j) {
					numOfNegativesInAttr++;
				}
			final += (double(numOfPositivesInAttr + numOfNegativesInAttr) / data.size()) * calcEntropy(&numOfPositivesInAttr, &numOfNegativesInAttr); //calculating the average information gain
		}
		return (initial + final); // calculating information gain
	}
	/*Selects best attribute*/
	int selectAttribute(vector<int> data) {

		double max = INT_MIN;
		int max_att;
		for (int i = 0; i < columnsForTrain; i++) {
			if (selected[i] == 0) {
				double infoGain = calcInfoGain(i, data);

				if (infoGain > max) {
					max = infoGain;
					max_att = i;
				}
			}
		}
		if (max == INT_MIN) {
			return -1;
		}
		selected[max_att] = 1;
		return max_att;
	}

	int choosePopularVote(vector<int> data) {
		int countYes = 0, countNo = 0;
		for (int i = 0; i < data.size(); ++i) {
			if (outcomeInput[data[i]] == 0) {
				countNo++;
			}
			else {
				countYes++;
			}
		}
		if (countYes > countNo) {
			return 1;
		}
		else {
			return 0;
		}
	}

	void makeDecision(vector<int> data, Node* node_passed) {
		bool isLeaf = true;
		if (data.size() == 0) {
			return;
		}
		for (int i = 1; i < data.size(); ++i) {
			if (outcomeInput[data[i]] != outcomeInput[data[i - 1]]) {
				isLeaf = false;
				break;
			}
		}
		if (isLeaf) {//indicates all data elements give same output 
			node_passed->answer = outcomeInput[data[0]];
			return;	//return the yes or no whichever is present
		}

		int selected_attribute = selectAttribute(data);

		if (selected_attribute != -1)
			node_passed->attribute = selected_attribute;

		if (selected_attribute == -1) {   //run out of attributes
			//going for popular choice
			node_passed->answer = choosePopularVote(data);
		}
		else {
			vector <vector<int>> sp(cardinality[selected_attribute]);//vector of vectors where each vector 
																	//represents the smaller data set after splitting

			for (int i = 0; i < data.size(); ++i) {
				sp[parsedData[selected_attribute][data[i]]].push_back(data[i]);
			}

			for (int i = 0; i < cardinality[selected_attribute]; ++i) {
				Node* child_node = createNode();
				node_passed->next[i] = child_node;
				makeDecision(sp[i], child_node);
			}
		}
	}
};

class DataParser {

public:
	map<int, vector<string>> dataset;

	void parseDataset( ifstream &file) {
	
		if (!dataset.empty()) {
			dataset.clear();
		}
		vector<string>vectorColumn[7];

		string line;

		//int i = 0;
		int rowCounter = 0;
		while (file) {
			rowCounter++;
			string line;
			getline(file, line); //reading one line from the file 
			stringstream lineStream(line); //wrapping string line in a stream so i can use getline to extract from the string
			string cell;
			int i = 0;
			while (getline(lineStream, cell, ',')) {
				vectorColumn[i].push_back(cell);
				i++;
			}
		} 
		file.close();

		for (int i = 0; i < columnsForTrain; ++i) {
			dataset.emplace(i, vectorColumn[i]);
		}
	}
	
	void parseColumns(int columnNumber, int rows) {
		map < string, int > mapx[10]; //array of maps; keys->each sample from dataset columns and values -> number of times this sample is contained in the dataset
									// each position of the array corresponds to the number of the column in the dataset
		for (int i =0; i < rows; i++) {

			/*For each sample for each column, tries to find if there is a key matching the same string
			
			If there is not such string in the map for this column, this means that it is a unique value and 
			and the iterator returns map[k].end().
			"count" is the serial number of the unique value, which is calculated by taking the size of the map at a specific index(columnNumber)
			
			If the there is such string, this means that it is not a specific value and we go to the next one.
			*/

				map< string, int >::iterator it = mapx[columnNumber].find(dataset[columnNumber][i]); 

				if (it == mapx[columnNumber].end()) {

					int count = mapx[columnNumber].size();
					mapx[columnNumber].insert(make_pair(dataset[columnNumber][i], count));
				}
			}
			cardinality.push_back(mapx[columnNumber].size());
			vector<int> inputData;
			for (int i = 0; i < rows; i++) {
				inputData.push_back(mapx[columnNumber][dataset[columnNumber][i]]);
			}
			parsedData.emplace(columnNumber, inputData); 
			mapx[columnNumber].clear();
	}
	void populateOutcome(ifstream& file){
		while (file) {

			string line;
			getline(file, line); //reading one line from the file 
			stringstream number(line);
			int x;
			number >> x;
			outcomeInput.push_back(x);
		}
		file.close();
	}
};

class ModelPrediction {
public:

	int correct = 0;
	int wrong = 0;

	int trueNegative = 0;
	int falseNegative = 0;

	int truePositive = 0;
	int falsePositive = 0;

	int num1 = 0;

	int predict(Node* node_passed, vector<int> test) {
		while (node_passed->answer == -1) {
			node_passed = node_passed->next[test[node_passed->attribute]];
		}
		return node_passed->answer;
	}

	void predict_interactive(Node* root, int x, vector<int>test_data) {

		int answer = predict(root, test_data);

		if (outcomeInput[x] == 1)
			num1++;

		if (answer == outcomeInput[x]) {
			correct++;
			if (outcomeInput[x] == 0) {
				trueNegative++;
			}
			else {
				truePositive++;
			}
		}
		else {
			wrong++;
			if (answer == 0) {
				falseNegative++;
			}
			else {
				falsePositive++;
			}
		}
	}
};

int main(){
	
	ifstream train_dataset("train_input.csv");
	ifstream test_dataset("test_input.csv");
	ifstream outcome_dataset("outcome_input.csv");

	DataParser *dataParser = new DataParser();
	dataParser->parseDataset(train_dataset);
	dataParser->populateOutcome(outcome_dataset);

	for (int i = 0; i < columnsForTrain; i++) {
		dataParser->parseColumns(i,rowsForTrain);
	}
	vector <int> v;
	for (int i = 0; i < rowsForTrain; i++)
		v.push_back(i);
	
	Node* root = createNode();
	DecisionTreeImp *tree = new DecisionTreeImp();
	
	tree->makeDecision(v, root);

  	parsedData.clear();
	dataParser->parseDataset(test_dataset);

	for (int i = 0; i < columnsForTest; i++) {
		dataParser->parseColumns(i,rowsForTest);
	}

	ModelPrediction* model = new ModelPrediction();
	for (int i = 0; i < rowsForTest; i++) {
		for (int j = 0; j < columnsForTest; j++) {
			model->predict_interactive(root, i, parsedData[j]);
		}
	}
		
	cout << "accuracy : " << double(model->correct) / (model->wrong + model->correct) * 100 << "%" << endl;
	cout << "precision for survived : " << double(model->truePositive) / (model->truePositive +model-> falsePositive) * 100 << "%" << endl;
	cout << "f1 score : " << (double(model->truePositive) /( model->truePositive + (model->falsePositive + model->falseNegative)/ 2))*100 << "%"<< endl;
}