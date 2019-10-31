/***************************************************************************
 *   Copyright (C) 2006 by Mian Zhou   *
 *   M.Zhou@reading.ac.uk   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef CVGABOR_H
#define CVGABOR_H

#include <iostream>


#include <cv.h>
#include <highgui.h>

#define PI 3.14159265
#define CV_GABOR_REAL 1
#define CV_GABOR_IMAG 2
#define CV_GABOR_MAG  3
#define CV_GABOR_PHASE 4

/**
@author Mian Zhou
*/
class CvGabor{
public:
    CvGabor();
    ~CvGabor();

  
     CvGabor(int iMu, int iNu, double dSigma);
     CvGabor(int iMu, int iNu, double dSigma, double dF);
     CvGabor(double dPhi, int iNu);
     CvGabor(double dPhi, int iNu, double dSigma);
     CvGabor(double dPhi, int iNu, double dSigma, double dF);
    bool IsInit();
    long mask_width();
    IplImage* get_image(int Type);
    bool IsKernelCreate();
    long get_mask_width();
    void Init(int iMu, int iNu, double dSigma, double dF);
    void Init(double dPhi, int iNu, double dSigma, double dF);
    void output_file(const char *filename, int Type);
    CvMat* get_matrix(int Type);
    void show(int Type);
    void conv_img(IplImage *src, IplImage *dst, int Type);
     CvGabor(int iMu, int iNu);
    void normalize( const CvArr* src, CvArr* dst, double a, double b, int norm_type, const CvArr* mask );
    void conv_img_a(IplImage *src, IplImage *dst, int Type);

protected:
    double Sigma;
    double F;
    double Kmax;
    double K;
    double Phi;
    bool bInitialised;
    bool bKernel;
    long Width;
    CvMat *Imag;
    CvMat *Real;
  
private:
    void creat_kernel();
    

};

#endif
