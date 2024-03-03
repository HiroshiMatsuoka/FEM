#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mkl_lapack.h>

/*1次元Poisson方程式を有限要素法で解く*/

/*問題
Ω=(0,1),
Γ1={0}
Γ2={1},
g1 = 0, g2 = 0, f = 1.0;
とする．この時，次を満たすu(x,y)を求めよ．
-△u = f in Ω, u(x,y) = g1 on Γ1, du/dx = 0 on Γ2,
*/


/* 方程式の非同次項 */
double f(double x){
  return 1.0;
}

int main(void){
  int ie, nnode, nelmt, nbc, i, j;
  double h;


  //総要素数
  nelmt = 100;
  //格子間隔
  h = 1.0 / nelmt;

  //節点の総数
  nnode = nelmt + 1;

  //メモリーの確保
  double x[nnode]; // 節点のx座標
  double fm[nnode]; // 境界条件を処理する前の要素自由項ベクトル
  double fe[nelmt]; // 境界条件を処理した後の要素自由項ベクトル
  double am[nnode][nnode]; // 境界条件を処理する前の要素係数行列
  double ae[nelmt][nelmt]; // 境界条件を処理した後の要素係数行列
  double aee[nelmt*nelmt]; // LAPACKで使用する要素係数行列

  double a = 0.0; //u(0) = a
  double b = 0.0; //u'(1) = b

  //LAPACで連立方程式を解く際に使う変数
  int X[nelmt];// 解ベクトル X
  int n = nelmt;
  int nrhs = 1;
  int ldab = nelmt;
  int ldb = n;
  int info ;

  #pragma omp parallel
  {
          //amの初期化
          #pragma omp ordered
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

              //節点の座標 (ここでは等分割する)
              for(ie = 0; ie < nnode; ie++){
                x[ie] = ie * h;
              }


      //要素係数行列の計算
      /*
      A_{00}^{k} =<L_{0},L_{0}>_{e_{k}}= 1 / (x_{k}-x_{k-1});
      A_{01}^{k} =<L_{1},L_{0}>_{e_{k}}= -1 / (x_{k}-x_{k-1});
      A_{10}^{k} =<L_{0},L_{1}>_{e_{k}}= -1 / (x_{k}-x_{k-1});
      A_{11}^{k} =<L_{1},L_{1}>_{e_{k}}= 1 / (x_{k}-x_{k-1});
      */

               // 対角成分
                am[0][0] = 1.0 / h;
                for(i=1; i<nelmt; i++){
                  am[i][i] = 1.0 / h + 1.0 / h;
                }
                am[nelmt][nelmt] = 1.0 / h;

                //帯
                for(i=0; i<nelmt; i++){
                  am[i][i+1] = -1.0 / h;
                  am[i+1][i] = -1.0 / h;
                }


              //要素自由項ベクトルの計算
                fm[0] = h * (2 * f(x[0]) + f(x[1])) / 6;
                for(i = 1; i < nnode; i++){
                   fm[i] = h * ((f(x[i-1]) + 2 * f(x[i])) + (2 * f(x[i]) + f(x[i+1]))) / 6;
                }
                fm[nelmt] = h * (f(x[nelmt-1]) + 2 * f(x[nelmt])) / 6 + b;


          //連立方程式の構成
          for(i=0; i<nelmt; i++){
            for(j=0; j<nelmt; j++){
              ae[i][j] = am[i+1][j+1];
            }
          }
        }



          for(i=0; i<nelmt; i++){
            for(j=0; j<nelmt; j++){
              aee[nelmt*i+j] = ae[i][j];
            }
          }


          //要素自由項ベクトル
          for(i=0; i<nelmt; i++){
            fe[i] = fm[i+1];
          }

          fe[0] = fe[0] + a / h;
  }

  // LAPACKのgesv関数を呼び出して連立方程式を解く
 dgesv_(&n, &nrhs, aee, &ldab, X, fe, &ldb, &info);

  // 解が得られたかどうかを確認
  if (info == 0) {
      // 解が得られた場合は、解ベクトル X を表示
      printf("0.000 %.30g\n", a);
      for(i=0; i<nelmt; i++){
      printf("%.20g %.30g\n", x[i+1], fe[i]);
    }
 } else {
      // 解が得られなかった場合は、エラーメッセージを表示
    printf("Failed to solve the system of equations.\n");
    printf("info = %d.\n", info);
 }


  return 0;
}
