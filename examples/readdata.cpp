// include files
// =============
#include "readdata.hpp"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cassert>
using std::cout;
using std::ios;
// =============

// =================================
// ======== Private Methods ========
// =================================

void
ReadData::eatCharacter_(char ch, fstream& fin) const
{
    char character;
    do
    {
        if(fin.peek() != ch )
            break;
        character = fin.get();
    }
    while (character == ch);
}


void
ReadData::collectData_(fstream& fin)
{
    char ch;
    char name[256];
    char value[256];

    // The purpose of this method is to read each line of
    // the data input file. Each line of this file may have
    // the following format:

    //    variable_name   =  value   # comments
    // example:
    //   reynolds   =  10000.0  # This is the Reynolds number


    // as we can see there is not restriction on the number
    // of spaces following or leading variable's name or
    // the equal sign. The purpose of this method is to
    // impose no restrictions at all on how long the comments
    // following '#' may be. To accomplish this, we follow the
    // steps described below.

    do
    {
        // first omit any spaces that might exist
        // before the name of the variable
        eatCharacter_(' ', fin);


        // then while the character picked from the input stream
        // is not a space(' '), equal sign('='), newline('\n') or
        // the character indicating the end of file(fin.eof())
        // read the name of the variable character by character
        int i = 0;
        while( (ch = fin.peek()) !=' ' && ch != '\t' && ch != '#' && ch != '=' && ch != '\n' && !fin.eof() )
        {
            fin.get(name[i++]);
        }

        // we shouldn't forget to insert the null character
        // at the end of the name of our char* because
        // otherwise we do not have a valid char*.
        name[i] = '\0';

        // now name contains the name of the variable before
        // the equal sign('='). We push it in the std::vector names_
        if (strlen(name) > 1){
            names_.push_back(string(name));
        }

        // now we go on dropping out from the input stream
        // any spaces before the '=', we drop '=' and any
        // spaces following it.

        eatCharacter_(' ', fin);
        eatCharacter_('=', fin);
        eatCharacter_(' ', fin);

        i=0;
        while( (ch = fin.peek()) !=' ' && ch != '\t' && ch != '#' && ch != '\n' && !fin.eof() )
        {
            fin.get(value[i++]);
        }

        // we insert the null character '\0' again to indicate
        // the end of the c-string type

        value[i] = '\0';

        // here we have the value of the first variable
        // and so we push it in the corresponding std::vector
        if (strlen(value) > 0) {
            values_.push_back(string(value));
        }


        // and we drop out of the input buffer all the other
        // characters (in our case comments) following the
        // assignment of the value of the variable until we find
        // the newline character '\n' or the end of file 'fin.eof()'

        while ( (ch = fin.peek()) != '\n' && !fin.eof() )
        {
            ch = fin.get();
        }

        // this last fin.get() is needed to extract the last character
        // from the input stream

        fin.get();

        // we are ready to move on to the next variable
    }
    while (!fin.eof());


    //cout << "names_.size(): " << names_.size() << std::endl;
    //cout << "values_.sie(): " << values_.size() << std::endl;
    assert(names_.size() == values_.size());
}

// =================================
// ======= Protected Methods =======
// =================================

void
ReadData::printVariable_(string name, const double variable) const
{
    cout << "  " << name << " = " << variable << "\n\n";
}

void
ReadData::printVariable_(string name, const float variable) const
{
    cout << "  " << name << " = " << variable << "\n\n";
}

void
ReadData::printVariable_(string name, const int variable) const
{
    cout << "  " << name << " = " << variable << "\n\n";
}

void
ReadData::printVariable_(string name, const long int  variable) const
{
    cout << "  " << name << " = " << variable << "\n\n";
}

void
ReadData::printVariable_(string name, const short variable) const
{
    cout << "  " << name << " = " << variable << "\n\n";
}

void
ReadData::printVariable_(string name, const char variable) const
{
    cout << "  " << name << " = " << variable << "\n\n";
}

void
ReadData::printVariable_(string name, const string variable) const
{
    cout << "  " << name << " = " << variable << "\n\n";
}

void
ReadData::printVariable_(string name, const bool variable) const
{
    cout << "  " << name << " = ";
    if (variable)
        cout << "yes\n\n";
    else
        cout << "no\n\n";
}


void
ReadData::initializeVariable_(string name, string& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();
    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);

    // now we know that name is the i_th entry of
    // vector names_. Hence its value is the string
    // values_[i]. Here we should go on calling
    // T operator on the string value[i]. But
    // we are not allowed to do this for several
    // types. For instance double, int, long int
    // bool do not know how to convert a string
    // to the corresponding type. Thus inevitably,
    // we will need template specializations for
    // those types.

    // So in the general case where we do not have
    // such restrictions we are done with:

    if (foundit)
    {
        variable = values_[i];
    }
}

void
ReadData::initializeVariable_(string name, float& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();

    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);

    // Here is what one should do in case of a double
    if (foundit)
    {
        variable = float(strtod(values_[i].c_str(), NULL));
    }
}

void
ReadData::initializeVariable_(string name, double& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();

    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);


    // Here is what one should do in case of a double
    if (foundit)
    {
        variable = double(strtod(values_[i].c_str(), NULL));
    }
}

void
ReadData::initializeVariable_(string name, short& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();

    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);


    // Here is what one should do in case of a double

    if (foundit)
    {
        variable = short(strtol(values_[i].c_str(), NULL, 0));
    }
}

void
ReadData::initializeVariable_(string name, int& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();

    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);

    // Here is what one should do in case of a double

    if (foundit)
    {
        variable = int(strtol(values_[i].c_str(), NULL, 0));

    }
}

void
ReadData::initializeVariable_(string name, long int& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();

    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);

    if (!foundit)
    {
        cout << "error in input data file\n";
        cout << "variable " << name << " was not found in data file\n";
        exit(1);
    }
    // Here is what one should do in case of a double

    if (foundit)
    {
        variable = strtol(values_[i].c_str(), NULL, 0);
    }
}

void
ReadData::initializeVariable_(string name, bool& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();

    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);


    // Here is what one should do in case of a double
    if (foundit)
    {
        if (values_[i] == "yes" || values_[i] == "true")
            variable = true;
        else if (values_[i] == "no" || values_[i] == "false")
            variable = false;
        else
        {
            cout << "value " << values_[i] << " was assigned to\n";
            cout << "boolean variable " << name << "\n";
            cout << "exiting ...\n";
            exit(1);
        }
    }
}

void
ReadData::initializeVariable_(string name, char& variable)
{
    // first we have to search in the vector names_
    // to find in which entry we have stored name
    int names_size = names_.size();

    int i = -1;
    bool foundit = false;
    bool finished;
    do
    {
        i++;
        if (names_[i] == name)
        {
            foundit = true;
        }
        finished = foundit || (i == names_size - 1);
    }
    while(!finished);

    // Here is what one should do in case of a char
    if (foundit)
    {
        const char* value = values_[i].c_str();
        variable = value[0];
    }
}

// =================================
// ========== Destructor ===========
// =================================

ReadData::~ReadData()
{
}

// =================================
// ========== Contructors ==========
// =================================

ReadData::ReadData()
{
    printparameters_ = false;
}

ReadData::ReadData(string datafile)
{
    init(datafile);
}

ReadData::ReadData(fstream& fin)
{
    init(fin);
}

// =================================
// ======== Public Methods =========
// =================================

void
ReadData::init(string datafile)
{
    printparameters_ = false;
    fstream fin(const_cast<char*>(datafile.c_str()), ios::in);
    if (!fin.is_open())
    {
        cout << "Couldn't open file: " << datafile << "\n";
        exit(1);
    }
    collectData_(fin);
    fin.close();
}

void
ReadData::init(fstream& fin)
{
    collectData_(fin);
}
