#ifndef readdata_hpp
#define readdata_hpp

// include files
// =============
#include <vector>
#include <string>
#include <fstream>
using std::fstream;
using std::vector;
using std::string;
// =============

class ReadData
{
private:
    vector<string> names_;
    vector<string> values_;

private:
    void printChar_(char ch) const;
    void eatCharacter_(char ch, fstream& fin) const;
    void collectData_(fstream& fin);
protected:
    bool printparameters_;

protected:
    void printVariable_(string name, const double variable) const;
    void printVariable_(string name, const int variable) const;
    void printVariable_(string name, const long int variable) const;
    void printVariable_(string name, const short variable) const;
    void printVariable_(string name, const float variable) const;
    void printVariable_(string name, const char variable) const;
    void printVariable_(string name, const bool variable) const;
    void printVariable_(string name, const string variable) const;

    void initializeVariable_(string name, double& variable);
    void initializeVariable_(string name, int& variable);
    void initializeVariable_(string name, long int& variable);
    void initializeVariable_(string name, short& variable);
    void initializeVariable_(string name, float& variable);
    void initializeVariable_(string name, char& variable);
    void initializeVariable_(string name, bool& variable);
    void initializeVariable_(string name, string& variable);

    virtual void printVariables_() const = 0;
    virtual void chooseDefaultParameters_() = 0;

public:
    virtual ~ReadData();
    ReadData();
    ReadData(string datafile);
    ReadData(fstream& fin);

    virtual void init(string datafile);
    virtual void init(fstream& fin);

};


#endif
