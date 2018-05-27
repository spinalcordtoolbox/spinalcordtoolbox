//
//  SCTemplate.h
//  SpinalCordDetectionValidation
//
//  Created by Benjamin De Leener on 2013-12-10.
//  Copyright (c) 2013 Benjamin De Leener. All rights reserved.
//

#ifndef __SpinalCordDetectionValidation__SCTemplate__
#define __SpinalCordDetectionValidation__SCTemplate__

#include "referential.h"

class SCTemplate
{
public:
    SCTemplate()
    {
        content_ = 0;
        size_[0] = 0; size_[1] = 0; size_[2] = 0;
    };
    virtual ~SCTemplate()
    {
        if (content_ != 0) {
            for (unsigned int x=0; x<size_[0]; x++) {
                for (unsigned int y=0; y<size_[1]; y++)
                    delete [] content_[x][y];
                delete [] content_[x];
            }
            delete [] content_;
        }
    };
    
    virtual void setSize(int* size)
    {
        if (content_ != 0) {
            for (unsigned int x=0; x<size_[0]; x++) {
                for (unsigned int y=0; y<size_[1]; y++)
                    delete [] content_[x][y];
                delete [] content_[x];
            }
            delete [] content_;
        }
        size_[0] = size[0]+(1-size[0]%2); size_[1] = size[1]+(1-size[1]%2); size_[2] = size[2]+(1-size[2]%2);
        
        content_ = new double**[size_[0]];
        for (unsigned int x=0; x<size_[0]; x++) {
            content_[x] = new double*[size_[1]];
            for (unsigned int y=0; y<size_[1]; y++) {
                content_[x][y] = new double[size_[2]];
                for (unsigned int z=0; z<size_[2]; z++)
                    content_[x][y][z] = 0.0;
            }
        }
    };
    virtual int* getSize() { return size_; };
    virtual void setContent(double*** content)
    {
        if (content_ != 0) {
            for (unsigned int x=0; x<size_[0]; x++) {
                for (unsigned int y=0; y<size_[1]; y++)
                    delete [] content_[x][y];
                delete [] content_[x];
            }
            delete [] content_;
        }
        content_ = content;
    };
    virtual void setContent(double*** content, int* size)
    {
        content_ = content;
        size_[0] = size[0]+(1-size[0]%2); size_[1] = size[1]+(1-size[1]%2); size_[2] = size[2]+(1-size[2]%2);
    };
    virtual double*** getContent() { return content_; };
    virtual double getData(int x, int y, int z) { return content_[x][y][z]; };
    virtual double& operator()(int x, int y, int z) { return content_[x][y][z]; };

protected:
    double*** content_;
    int size_[3]; // always impair
    
    Referential refCourant_;
};

#endif /* defined(__SpinalCordDetectionValidation__SCTemplate__) */
