#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mkl_lapack.h>


/*2次元Poisson方程式を有限要素法で解く*/

/*問題
Ω=(0,1)×(0,1),
Γ1={(x,y)|x=0, 0<=y<=1}U{(x,y)|0<=x<=1, y=0}
Γ2={(x,y)|x=1, 0<y<=1}U{(x,y)|0<x<=1, y=1},
g1 = 0, g2 = 0, f = 24.0とする．この時，次を満たすu(x,y)を求めよ．
-△u = f in Ω, u(x,y) = g1 on Γ1,  ∂u/∂n = 0 on Γ2,
ただし, nは、Ωの境界上の単位法線ベクトルとする.
*/

// 要素の形は "fem-element.pdf" に記載




//double f(double, double);

/* 方程式の非同次項 */
double f(double x, double y){
  return 24.0;
}

//boundary condition
double g(double x, double y){
  return 0;

}

int main(void){
  int ie, nnode, nelmt, nbc, i, j;
  double h;


  //総要素数
  nelmt = 800;
  int ngrisp;
  // 格子間隔数
  ngrisp = sqrt(nelmt / 2);
  // 格子間隔
  h = 1.0 / ngrisp;

  //節点の総数
  nnode = (ngrisp + 1) * (ngrisp + 1) - 1;

  // ループで計算する節点の数
  int N;
  N = ngrisp * ngrisp + ngrisp - 2;

  //メモリーの確保
  double x[nnode]; // 節点のx座標
  double xx[nnode]; //実行結果を表示する際に活用する変数
  double y[nnode]; //節点のy座標
  double yy[nnode]; //実行結果を表示する際に活用する変数
  double fe[nnode]; //要素自由項ベクトル
  double am[nnode][nnode]; // 境界条件を処理する前の要素係数行列
  double ae[nnode][nnode]; // 境界条件を処理した後の要素係数行列
  double fmI[nnode];   // 要素type-1での要素自由項ベクトル
  double fmII[nnode];  // 要素type-2での要素自由項ベクトル　
  double aee[nelmt*nelmt]; // LAPACKで使用する要素係数行列

  //LAPACで連立方程式を解く際に使う変数
  int X[nelmt];   // 解ベクトル X
  int n = nnode;
  int nrhs = 1;
  int ldab = nnode;
  int ldb = n;
  int info ;

  // 係数の設定・・・理論的に導出済み
  //要素type-1での係数行列を求めるための行列
  double K1[3][3];
  double AI[3][3];
  K1[0][0] = 0.0; K1[0][1] = 0.0; K1[0][2] = 0.0;
  K1[1][0] = (-1.0)*h; K1[1][1] = 1.0*h; K1[1][2] = 0.0;
  K1[2][0] = 0.0; K1[2][1] = (-1.0)*h; K1[2][2] = 1.0*h;

  //要素type-2での係数行列を求めるための行列
  double K2[3][3];
  double AII[3][3];
  K2[0][0] = 0.0; K2[0][1] = 0.0; K2[0][2] = 0.0;
  K2[1][0] = 0; K2[1][1] = 1.0*h; K2[1][2] = (-1.0)*h;
  K2[2][0] = (-1.0)*h; K2[2][1] = 0; K2[2][2] = 1.0*h;


  #pragma omp parallel
  {
          //amの初期化
      #pragma omp for nowait ordered
      {
        for(i=0; i<nnode; i++){
            for(j=0; j<nnode; j++){
            am[i][j] = 0;
          }
        }

          //aeの初期化
          for(i=0; i<nelmt; i++){
            for(j=0; j<nelmt; j++){
             ae[i][j] = 0;
           }
         }
      }

      //節点の座標 (ここでは等分割する)
     #pragma omp ordered
      {  for(ie = 0; ie < ngrisp+1; ie++){
            x[ie] = ie * h;
            y[ie] = ie * h;
          }
      }
      //実行結果を表示する際に活用する変数　xx, yy
      for(i=0; i<ngrisp+1; i++){
        for(j=0; j<ngrisp+1; j++){
          xx[i*(ngrisp+1)+j] = x[i];
        }
      }

      for(i=0; i<nnode; i++){
        yy[i] = y[i % (ngrisp+1)];
      }
  }


  //要素係数行列の計算
  /*
  A_{00}^{k} =<L_{0},L_{0}>_{e_{k}}= 1 / (x_{k}-x_{k-1});
  A_{01}^{k} =<L_{1},L_{0}>_{e_{k}}= -1 / (x_{k}-x_{k-1});
  A_{10}^{k} =<L_{0},L_{1}>_{e_{k}}= -1 / (x_{k}-x_{k-1});
  A_{11}^{k} =<L_{1},L_{1}>_{e_{k}}= 1 / (x_{k}-x_{k-1});
  */

  #pragma omp parallel
  {
        //各要素の係数行列の設定
        #pragma omp ordered
        {
            for(i=0; i<3; i++){
              for(j=0; j<3; j++){
                AI[i][j] = 0.5 * (K1[1][i] * K1[1][j] + K1[2][i] * K1[2][j]);
              }
            }

            //type-2
            for(i=0; i<3; i++){
              for(j=0; j<3; j++){
                AII[i][j] = 0.5 * (K2[1][i] * K2[1][j] + K2[2][i] * K2[2][j]);
              }
            }
        }

        //全ての要素type-1での係数行列の和
        for(i=0; i<N+1; i++){
          if((i-ngrisp)%(ngrisp+1)!=0){
          am[i][i] += AI[0][0];
          am[i][i+ngrisp+1] += AI[0][1];
          am[i][i+ngrisp+2] += AI[0][2];
          am[i+ngrisp+1][i] += AI[1][0];
          am[i+ngrisp+1][i+ngrisp+1] += AI[1][1];
          am[i+ngrisp+1][i+ngrisp+2] += AI[1][2];
          am[i+ngrisp+2][i] += AI[2][0];
          am[i+ngrisp+2][i+ngrisp+1] += AI[2][1];
          am[i+ngrisp+2][i+ngrisp+2] += AI[2][2];
          }
        }

        //全ての要素type-2での係数行列の和
        for(i=0; i<N+1; i++){
          if((i-ngrisp)%(ngrisp+1)!=0){
          am[i][i] += AII[0][0];
          am[i][i+ngrisp+2] += AII[0][1];
          am[i][i+1] += AII[0][2];
          am[i+ngrisp+2][i] += AII[1][0];
          am[i+ngrisp+2][i+ngrisp+2] += AII[1][1];
          am[i+ngrisp+2][i+1] += AII[1][2];
          am[i+1][i] += AII[2][0];
          am[i+1][i+ngrisp+2] += AII[2][1];
          am[i+1][i+1] += AII[2][2];
        }
        }

      //要素自由項ベクトルの計算
         for(i=0; i<N+1; i++){
            if((i-ngrisp)%(ngrisp+1)!=0){
            fmI[i] = h * h * (2 * (f(xx[i], yy[i])) + f(xx[i+ngrisp+1], yy[i]) + f(xx[i+ngrisp+1], yy[i+ngrisp+2])) / 24;
            fmI[i+ngrisp+1] = h * h * (f(xx[i], yy[i]) + 2 * (f(xx[i+ngrisp+1], yy[i])) + f(xx[i+ngrisp+1], yy[i+ngrisp+2])) / 24;
            fmI[i+ngrisp+2] = h * h * (f(xx[i], yy[i]) + f(xx[i+ngrisp+1], yy[i]) + 2 * (f(xx[i+ngrisp+1], yy[i+ngrisp+2]))) / 24;
            fmII[i] = h * h * (2 * f(xx[i], yy[i]) + f(xx[i+ngrisp+2], yy[i+1]) + f(xx[i], y[i+1])) / 24;
            fmII[i+ngrisp+2] = h * h * (f(xx[i], yy[i]) + f(xx[i+ngrisp+2], yy[i+1]) + 2 * f(xx[i], yy[i+1])) / 24;
            fmII[i+1] = h * h * (f(xx[i], yy[i]) + f(xx[i+ngrisp+2], yy[i+1]) + 2 * f(xx[i], yy[i+1])) / 24;
            fe[i] += fmI[i] + fmII[i];
            fe[i+ngrisp+1] += fmI[i+ngrisp+1];
            fe[i+ngrisp+2] += fmI[i+ngrisp+2] + fmII[i+ngrisp+2];
            fe[i+1] += fmII[i+1];
            }
          }
  }

  // 境界条件処理
   // vi=g1(i=k,k*(ngrisp+1)(k=1,2,...,ngrisp))
   //generate ae
   #pragma omp parallel
   {
           for(i=0; i<nnode; i++){
             for(j=0; j<nnode; j++){
               ae[i][j] = am[i][j];
             }
           }

           // boundary condition for element matrix
           for(i=0; i<ngrisp+1; i++){
             for(j=0; j<nnode; j++){
                 ae[i][i] = 1;
                 ae[i][j] = 0;
                 ae[j][i] = 0;
               }
             }

           for(i=0; i<ngrisp+1; i++){
              for(j=0; j<nnode; j++){
                ae[i*(ngrisp+1)][i*(ngrisp+1)] = 1;
                ae[i*(ngrisp+1)][j] = 0;
                ae[j][i*(ngrisp+1)] = 0;
            }
          }
          // boundary condition for element free vector
          for(i=0; i<ngrisp+1; i++){
            fe[i] = g(0.0, yy[i]);
            fe[i*(ngrisp+1)] = g(xx[i*(ngrisp+1)], 0.0);
          }
  }
  //連立方程式の構成
  for(i=0; i<nnode; i++){
    for(j=0; j<nnode; j++){
      aee[nnode*i+j] = ae[i][j];
    }
  }

  // LAPACKのgesv関数を呼び出して連立方程式を解く
  dgesv_(&n, &nrhs, aee, &ldab, X, fe, &ldb, &info);

  // 解が得られるか確認
  if (info == 0) {
      // 解が得られる場合は、解を表示
          for(i=0; i<nnode; i++){
          printf("%.20g  %.20g  %.30g\n", xx[i], yy[i], fe[i]);
        }
   } else {
            // 解が得られない場合は、メッセージを表示
            printf("Failed to solve the system of equations.\n");
            printf("info = %d.\n", info);
           }


  return 0;
}
