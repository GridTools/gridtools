#ifndef sortingsearching_hpp
#define sortingsearching_hpp

#include <iostream>
using std::cout;
template<typename S>
void heapsort (int n, S *v)
{
    if (n < 2) return;
    --v;    
    int i,j;
    int l = (n >> 1) + 1;
    int k = n;
    S rv;
    for (;;) 
    {
        if (l > 1)
            rv = v[--l];
        else 
        {
            rv = v[k];
            v[k] = v[1];
            if (--k == 1) 
            {
                v[1] = rv;
                return;
            }
        }
        i = l;
        j = l << 1;
        while (j <= k) 
        {
            if (j < k && v[j] < v[j+1]) ++j;
            if (rv < v[j]) 
            {
                v[i] = v[j];
                j += (i = j);
            } 
            else 
                j = k + 1;
        }
        v[i] = rv;
    }
}

template<typename S, typename F>
void heapsort (int n, S* sortvec, F* followvec)
{
    if (n < 2) return;
    --sortvec;
    --followvec;
    
    int i,j;
    int l = (n >> 1) + 1;
    int k = n;
    S rv;
    F rf;
    for (;;) 
    {
        if (l > 1)
        {
            rv = sortvec[--l];
            rf = followvec[l];
        }
        else 
        {
            rv = sortvec[k];
            rf = followvec[k];
            sortvec[k] = sortvec[1];
            followvec[k] = followvec[1];
            if (--k == 1) 
            {
                sortvec[1] = rv;
                followvec[1] = rf;
                return;
            }
        }
        
        i = l;
        j = l << 1;
        while (j <= k) 
        {
            if (j < k && sortvec[j] < sortvec[j+1]) ++j;
            if (rv < sortvec[j]) 
            {
                sortvec[i] = sortvec[j];
                followvec[i] = followvec[j];
                j += (i = j);
            } 
            else 
                j = k + 1;
        }
        sortvec[i] = rv;
        followvec[i] = rf;
    }
}

template <typename T> inline void 
SWAP(T& a, T& b)
{
  T temp = a;
  a = b;
  b = temp;
}

template <typename S>
void nrSort(int n, S* arr)
{
    arr--;
    int i,j,k,l=1;
    int ir=n;
    int* istack;
    int jstack=0;
    S a;
    
    const int M = 7;
    const int NSTACK=50;
    
    istack = new int[NSTACK+1];
    
    for (;;) {
        if (ir-l < M) {
            for (j=l+1;j<=ir;j++) {
                a=arr[j];
                for (i=j-1;i>=l;i--) {
                    if (arr[i] <= a) break;
                    arr[i+1]=arr[i];
                }
                arr[i+1]=a;
            }
            if (jstack == 0) break;
            ir=istack[jstack--];
            l=istack[jstack--];
        } else {
            k=(l+ir) >> 1;
            SWAP(arr[k],arr[l+1]);
            
            if (arr[l] > arr[ir]) 
            {
                SWAP(arr[l],arr[ir]);
            }
            if (arr[l+1] > arr[ir]) 
            {
                SWAP(arr[l+1],arr[ir]);
            }
            if (arr[l] > arr[l+1]) 
            {
                SWAP(arr[l],arr[l+1]);
            }
            i=l+1;
            j=ir;
            a=arr[l+1];
            for (;;) {
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i) break;
                SWAP(arr[i],arr[j]);
            }
            arr[l+1]=arr[j];
            arr[j]=a;
            jstack += 2;
            if (jstack >= NSTACK) cout << "NSTACK too small in nrSort.\n";
            if (ir-i+1 >= j-l) {
                istack[jstack]=ir;
                istack[jstack-1]=i;
                ir=j-1;
                
            } else {
                istack[jstack]=j-1;
                istack[jstack-1]=l;
                l=i;
            }
        }
    }
    delete[] istack;
}

template <typename S, typename F>
void nrSort(int n, S* arr, F* brr)
{
    arr--;
    brr--;

    const int M = 7;
    const int NSTACK=50;
    int i,ir=n,j,k,l=1;
    int* istack;
    int jstack=0;
    S a;
    F b;
    
    istack = new S[NSTACK+1];
    for (;;) {
        if (ir-l < M) {
            for (j=l+1;j<=ir;j++) {
                a=arr[j];
                b=brr[j];
                for (i=j-1;i>=l;i--) {
                    if (arr[i] <= a) break;
                    arr[i+1]=arr[i];
                    brr[i+1]=brr[i];
                }
                arr[i+1]=a;
                brr[i+1]=b;
            }
            if (jstack == 0) break;
            
            ir=istack[jstack--];
            l=istack[jstack--];
        } else {
            k=(l+ir) >> 1;
            SWAP(arr[k],arr[l+1]);
            SWAP(brr[k],brr[l+1]);
            if (arr[l] > arr[ir]) {
                SWAP(arr[l],arr[ir]);
                SWAP(brr[l],brr[ir]);
            }
            if (arr[l+1] > arr[ir]) {
                SWAP(arr[l+1],arr[ir]);
                SWAP(brr[l+1],brr[ir]);
            }
            if (arr[l] > arr[l+1]) {
                SWAP(arr[l],arr[l+1]);
                SWAP(brr[l],brr[l+1]);
            }
            i=l+1;
            j=ir;
            
            a=arr[l+1];
            b=brr[l+1];
            for (;;) {
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i) break;
                SWAP(arr[i],arr[j]);
                SWAP(brr[i],brr[j]);
            }
            arr[l+1]=arr[j];
            arr[j]=a;
            brr[l+1]=brr[j];
            brr[j]=b;
            jstack += 2;
            if (jstack >= NSTACK) cout << "NSTACK too small in nrSort with follower\n";
            if (ir-i+1 >= j-l) {
                istack[jstack]=ir;
                istack[jstack-1]=i;
                ir=j-1;
            } else {
                istack[jstack]=j-1;
                istack[jstack-1]=l;
                l=i;
            }
        }
    }
}       

template<typename S>
int binsearch(S x, S* vec, int n)
{
    int low,high,mid;
    low = 0;
    high = n-1;
    
    while(low <= high)
    {
        mid = (low + high)/2;
        if(x < vec[mid]) 
            high = mid - 1;
        else if ( x > vec[mid] )
            low = mid + 1;
        else   /* found the value */
            return mid;
    }
    return -1; /* nothing */
}

template<typename S>
int seqsearch(S x, S* vec, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        if (vec[i] == x)
            return i;
    }
    return -1; /* nothing */
}

#endif
