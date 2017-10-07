#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */


  // Laser measurement covariance matrix
  R_lidar_ = MatrixXd(2,2);

  // Radar measurement covariance matrix
  R_radar_ = MatrixXd(3,3);
  
  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;
  
  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);

  // Calculate weight matrix
  weights_ = VectorXd(2*n_aug_ + 1);
  
  // NIS for radar
  NIS_radar_ = 0.0;

  // NIS for laser
  NIS_lidar_ = 0.0;

  // Initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // Time when the state is true, in us
  time_us_ = 0.0;

  // Init measurement covariance matrix
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  // Set weight matrix
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_ + 1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_ + lambda_);
    weights_(i) = weight;
  }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  double dt;
  double ro, theta, ro_dot;
  double px, py;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if(!is_initialized_) {
    /**
      * Initialize the state x_ with the first measurement.
      * Create the covariance matrix.
    */
    // init state
    x_ << 0, 0, 0, 0, 0;

    // init state covariance matrix
    P_ = MatrixXd::Identity(5,5);

    if((use_radar_)&&
      (meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
      /**
      Initialize state and state covariance matrix of radar.
      */

      ro = meas_package.raw_measurements_[0];
      theta = meas_package.raw_measurements_[1];
      ro_dot = meas_package.raw_measurements_[2];

      //Set the state with initial location and zero velocity,
      //for radar measurement does not contain enough information to determine
      //the state variable velocities v​x and v​y.
      px = ro * sin(theta);
      py = ro * cos(theta);

      x_(0) = px;
      x_(1) = py; 
      x_(3) = theta;

      //state covariance matrix P
      P_(0,0) = 0.3;
      P_(1,1) = 0.3;
      
      time_us_ = meas_package.timestamp_;
    }

    else if((use_laser_)&&
           (meas_package.sensor_type_ == MeasurementPackage::LASER)) {
      /**
      Initialize state and state covariance matrix of laser.
      */

      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];

      //Set the state with initial location and zero velocity.
      x_(0) = px;
      x_(1) = py;

      //state covariance matrix P
      P_(0,0) = 0.15;
      P_(1,1) = 0.15;

      time_us_ = meas_package.timestamp_;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

   if((use_radar_)&&
   (meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
     // Radar updates
     UpdateRadar(meas_package);
   }
   else if((use_laser_)&&
   (meas_package.sensor_type_ == MeasurementPackage::LASER)) {
     // Lidar updates
     UpdateLidar(meas_package);
   }
   else{
     cout << "Error - No type of sensor" << endl;
   }

   //print the output
   cout << "x= " << x_.transpose() << endl;
   //cout << "P= " << P_ << endl;
   cout << "-------------" << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  double px, py, v, yaw, yawd, nu_a, nu_yawdd;
  double px_pred, py_pred, v_pred, yaw_pred, yawd_pred;
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_ + 1);
  MatrixXd L;
  VectorXd x_pred = VectorXd(n_x_);
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  VectorXd x_diff = VectorXd(n_x_);

  /******************************
   * Step1: Generate Sigma Points
  ******************************/

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  //create augmented state covariance
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;

  //create sigma points matrix
  Xsig_aug.col(0) = x_aug;
  L = P_aug.llt().matrixL();
  for(int i=0; i<n_aug_; i++) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  /*****************************
   * Step2: Predict Sigma Points
  *****************************/

  for(int i=0; i<2*n_aug_ + 1; i++) {
    
    //extract state values for calculating
    px = Xsig_aug(0,i);
    py = Xsig_aug(1,i);
    v = Xsig_aug(2,i);
    yaw = Xsig_aug(3,i);
    yawd = Xsig_aug(4,i);
    nu_a = Xsig_aug(5,i);
    nu_yawdd = Xsig_aug(6,i);

    //predict state values for CTRV model
    if(fabs(yawd) > 0.0001) {
      px_pred = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_pred = py + v/yawd * (-cos(yaw + yawd*delta_t) + cos(yaw));
    }
    else {
      px_pred = px + v*cos(yaw)*delta_t;
      py_pred = py + v*sin(yaw)*delta_t;
    }
    v_pred = v;
    yaw_pred = yaw + yawd*delta_t;
    yawd_pred = yawd;

    //add noise
    px_pred = px_pred + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
    py_pred = py_pred + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
    v_pred = v_pred + delta_t*nu_a;
    yaw_pred = yaw_pred + 0.5*delta_t*delta_t*nu_yawdd;
    yawd_pred = yawd_pred + delta_t*nu_yawdd;

    //update prdicted sigma point matrix
    Xsig_pred_(0,i) = px_pred;
    Xsig_pred_(1,i) = py_pred;
    Xsig_pred_(2,i) = v_pred;
    Xsig_pred_(3,i) = yaw_pred;
    Xsig_pred_(4,i) = yawd_pred;
  }

  /************************************
   * Step3: Predict Mean and Covariance
  ************************************/

  //predict state mean
  x_pred.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {
    x_pred = x_pred + weights_(i) * Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_pred.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {
    //state difference
    x_diff = Xsig_pred_.col(i) - x_pred;
    //angle normalization for yaw
    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) <-M_PI) x_diff(3) += 2.*M_PI;

    P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose();
  }

  //write result
  x_ = x_pred;
  P_ = P_pred;

}//end of UKF::Prediction()

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  double px, py;
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_pred = VectorXd(n_z);
  VectorXd z_diff;
  VectorXd x_diff;
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_ + 1);
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  MatrixXd K;

  /****************************
   * Step1: Predict Measurement 
  ****************************/

  //transform sigma points into lidar measurement space
  for(int i=0; i<2*n_aug_ + 1; i++) {
    //extract state values for calculating
    px = Xsig_pred_(0,i);
    py = Xsig_pred_(1,i);

    //measurement model
    Zsig(0,i) = px;
    Zsig(1,i) = py;
  }

  //predict measurement mean
  z_pred.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //predict measurement covariance matrix S
  S.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {
    //measurement difference
    z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_lidar_;

  /****************************
   * Step2: Update State 
  ****************************/

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {

    //measurement difference
    z_diff = Zsig.col(i) - z_pred;
    //state difference
    x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K
  K = Tc * S.inverse();

  //measurement difference update
  z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //calculate NIS
  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;

  //print the output
  cout << "Sensor type: " << meas_package.sensor_type_ << endl;
  cout << "z= " << z.transpose() << endl;
  cout << "z_pred= " << z_pred.transpose() << endl;
  //cout << "S= " << S << endl;
  cout << "NIS= " << NIS_lidar_ << endl;

}//end of UKF::UpdateLidar

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  int n_z = 3;
  double px, py, v, yaw, yawd;
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_pred = VectorXd(n_z);
  VectorXd z_diff;
  VectorXd x_diff;
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_ + 1);
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd R = MatrixXd(n_z, n_z);
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  MatrixXd K;

  /****************************
   * Step1: Predict Measurement 
  ****************************/

  //transform sigma points into radar measurement space
  for(int i=0; i<2*n_aug_ + 1; i++) {
    //extract state values for calculating
    px = Xsig_pred_(0,i);
    py = Xsig_pred_(1,i);
    v = Xsig_pred_(2,i);
    yaw = Xsig_pred_(3,i);

    //measurement model
    Zsig(0,i) = sqrt(px*px + py*py);
    Zsig(1,i) = atan2(py,px);
    Zsig(2,i) = (px*cos(yaw)*v + py*sin(yaw)*v)/sqrt(px*px + py*py);
  }

  //predict measurement mean
  z_pred.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //predict measurement covariance matrix S
  S.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {
    //measurement difference
    z_diff = Zsig.col(i) - z_pred;
    //angle normalization for theta
    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_radar_;

  /****************************
   * Step2: Update State 
  ****************************/

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i=0; i<2*n_aug_ + 1; i++) {

    //measurement difference
    z_diff = Zsig.col(i) - z_pred;
    //angle normalization for theta
    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

    //state difference
    x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization for yaw
    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K
  K = Tc * S.inverse();

  //measurement difference update
  z_diff = z - z_pred;
  //angle normalization for theta
  while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  //print the output
  cout << "Sensor type: " << meas_package.sensor_type_ << endl;
  cout << "z= " << z.transpose() << endl;
  cout << "z_pred= " << z_pred.transpose() << endl;
  //cout << "S= " << S << endl;
  cout << "NIS= " << NIS_radar_ << endl;

}//end of UKF::UpdateRadar


