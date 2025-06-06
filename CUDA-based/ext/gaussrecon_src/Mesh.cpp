/*
Copyright (c) 2018, Wenjia Lu, Zuoqiang Shi, Jian Sun and Bin Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of Tsinghua University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "Mesh.h"

CoredVectorMeshData::CoredVectorMeshData( void ) { oocPointIndex = polygonIndex = 0; }
void CoredVectorMeshData::resetIterator ( void ) { oocPointIndex = polygonIndex = 0; }
int CoredVectorMeshData::addOutOfCorePoint(const Point& p){
	oocPoints.push_back(p);
	return int(oocPoints.size())-1;
}
int CoredVectorMeshData::addPolygon( const vector< CoredVertexIndex >& vertices )
{
	std::vector< int > polygon( vertices.size() );
	for( int i=0 ; i<vertices.size() ; i++ ) 
		if( vertices[i].inCore ) polygon[i] =  vertices[i].idx;
		else                     polygon[i] = -vertices[i].idx-1;
		polygons.push_back( polygon );
		return int( polygons.size() )-1;
}
int CoredVectorMeshData::nextOutOfCorePoint(Point& p){
	if(oocPointIndex<int(oocPoints.size())){
		p=oocPoints[oocPointIndex++];
		return 1;
	}
	else{return 0;}
}
int CoredVectorMeshData::nextPolygon( vector< CoredVertexIndex >& vertices )
{
	if( polygonIndex<int( polygons.size() ) )
	{
		std::vector< int >& polygon = polygons[ polygonIndex++ ];
		vertices.resize( polygon.size() );
		for( int i=0 ; i<polygon.size() ; i++ )
			if( polygon[i]<0 ) vertices[i].idx = -polygon[i]-1 , vertices[i].inCore = false;
			else               vertices[i].idx =  polygon[i]   , vertices[i].inCore = true;
			return 1;
	}
	else return 0;
}
int CoredVectorMeshData::outOfCorePointCount(void){return int(oocPoints.size());}
int CoredVectorMeshData::polygonCount( void ) { return int( polygons.size() ); }

