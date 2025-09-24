import React, { useState, useEffect } from "react";
import "./App.css";
import * as ReactRouterDOM from "react-router-dom";

import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";

import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Alert, AlertDescription } from "./components/ui/alert";
import { Badge } from "./components/ui/badge";
import { TrendingUp, DollarSign, Users, AlertTriangle, CheckCircle, XCircle, BarChart3, Calculator, Database } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Professional Banking Color Palette
const colors = {
  primary: '#1e40af',
  secondary: '#3b82f6', 
  success: '#22c55e',
  warning: '#f59e0b',
  danger: '#ef4444',
  background: '#f8fafc',
  surface: '#ffffff',
  text: '#1e293b',
  muted: '#64748b'
};

const COLORS = ['#1e40af', '#3b82f6', '#6366f1', '#8b5cf6'];

// Navigation Component
const Navigation = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: BarChart3 },
    { path: '/predict', label: 'Loan Predictor', icon: Calculator },
    { path: '/data', label: 'Data Analysis', icon: Database }
  ];

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            <h1 className="text-xl font-bold text-gray-900">LoanAnalyzer Pro</h1>
          </div>
          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                    isActive 
                      ? 'bg-blue-100 text-blue-700 font-medium' 
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

// Dashboard Component
const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, perfRes] = await Promise.all([
        axios.get(`${API}/dataset/stats`),
        axios.get(`${API}/model/performance`)
      ]);
      setStats(statsRes.data);
      setPerformance(perfRes.data);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!stats || !performance) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
        <p className="text-gray-600">Failed to load dashboard data</p>
      </div>
    );
  }

  // Prepare chart data
  const incomeDistData = [
    { name: 'Low Income (<$3K)', value: stats.feature_distributions.income_distribution.low, color: '#ef4444' },
    { name: 'Medium Income ($3K-$7K)', value: stats.feature_distributions.income_distribution.medium, color: '#f59e0b' },
    { name: 'High Income (>$7K)', value: stats.feature_distributions.income_distribution.high, color: '#22c55e' }
  ];

  const creditHistoryData = [
    { name: 'Good Credit', value: stats.feature_distributions.credit_history_distribution['1'], color: '#22c55e' },
    { name: 'Poor Credit', value: stats.feature_distributions.credit_history_distribution['0'], color: '#ef4444' }
  ];

  const loanAmountData = [
    { name: 'Small (<$100K)', value: stats.feature_distributions.loan_amount_distribution.small },
    { name: 'Medium ($100K-$200K)', value: stats.feature_distributions.loan_amount_distribution.medium },
    { name: 'Large (>$200K)', value: stats.feature_distributions.loan_amount_distribution.large }
  ];

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Records</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_records.toLocaleString()}</p>
              </div>
              <Database className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Default Rate</p>
                <p className="text-2xl font-bold text-red-600">{(stats.default_rate * 100).toFixed(1)}%</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Income</p>
                <p className="text-2xl font-bold text-green-600">${stats.avg_income.toLocaleString()}</p>
              </div>
              <DollarSign className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Model Accuracy</p>
                <p className="text-2xl font-bold text-blue-600">{(performance.accuracy * 100).toFixed(1)}%</p>
              </div>
              <CheckCircle className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Income Distribution</CardTitle>
            <CardDescription>Distribution of applicant income levels</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={incomeDistData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {incomeDistData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Credit History Impact</CardTitle>
            <CardDescription>Credit history distribution in dataset</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={creditHistoryData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {creditHistoryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Loan Amount Distribution</CardTitle>
            <CardDescription>Distribution of requested loan amounts</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={loanAmountData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
            <CardDescription>Key model metrics and information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Model Type:</span>
              <Badge variant="secondary">{performance.model_type}</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Training Samples:</span>
              <span className="font-medium">{performance.total_samples.toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Accuracy Score:</span>
              <span className="font-medium text-green-600">{(performance.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Dataset Default Rate:</span>
              <span className="font-medium text-red-600">{(performance.default_rate * 100).toFixed(2)}%</span>  
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Loan Prediction Component
const LoanPredictor = () => {
  const [formData, setFormData] = useState({
    applicant_income: '',
    coapplicant_income: '',
    loan_amount: '',
    loan_amount_term: '360',
    credit_history: '1',
    gender: '',
    married: '',
    dependents: '',
    education: '',
    self_employed: '',
    property_area: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }));
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API}/predict`, {
        applicant_income: parseFloat(formData.applicant_income),
        coapplicant_income: parseFloat(formData.coapplicant_income) || 0,
        loan_amount: parseFloat(formData.loan_amount),
        loan_amount_term: parseInt(formData.loan_amount_term),
        credit_history: parseInt(formData.credit_history),
        gender: formData.gender,
        married: formData.married,
        dependents: formData.dependents,
        education: formData.education,
        self_employed: formData.self_employed,
        property_area: formData.property_area
      });

      setPrediction(response.data);
    } catch (error) {
      setError('Error making prediction. Please check your inputs and try again.');
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'Low': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'High': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getPredictionIcon = (predictedClass) => {
    return predictedClass === 'Approved' ? 
      <CheckCircle className="h-6 w-6 text-green-600" /> : 
      <XCircle className="h-6 w-6 text-red-600" />;
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Loan Default Prediction</h2>
        <p className="text-gray-600">Enter applicant details to assess loan default risk</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Loan Application Form</CardTitle>
            <CardDescription>Fill in all required fields for accurate prediction</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="gender">Gender</Label>
                  <Select onValueChange={(value) => handleInputChange('gender', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select gender" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Male">Male</SelectItem>
                      <SelectItem value="Female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="married">Married</Label>
                  <Select onValueChange={(value) => handleInputChange('married', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Marital status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Yes">Yes</SelectItem>
                      <SelectItem value="No">No</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="dependents">Dependents</Label>
                  <Select onValueChange={(value) => handleInputChange('dependents', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Number of dependents" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">0</SelectItem>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                      <SelectItem value="3+">3+</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="education">Education</Label>
                  <Select onValueChange={(value) => handleInputChange('education', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Education level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Graduate">Graduate</SelectItem>
                      <SelectItem value="Not Graduate">Not Graduate</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="self_employed">Self Employed</Label>
                  <Select onValueChange={(value) => handleInputChange('self_employed', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Employment type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Yes">Yes</SelectItem>
                      <SelectItem value="No">No</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="property_area">Property Area</Label>
                  <Select onValueChange={(value) => handleInputChange('property_area', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Property location" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Urban">Urban</SelectItem>
                      <SelectItem value="Semiurban">Semiurban</SelectItem>
                      <SelectItem value="Rural">Rural</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="applicant_income">Applicant Income (Annual)</Label>
                <Input
                  type="number"
                  placeholder="e.g., 50000"
                  value={formData.applicant_income}
                  onChange={(e) => handleInputChange('applicant_income', e.target.value)}
                  required
                />
              </div>

              <div>
                <Label htmlFor="coapplicant_income">Co-applicant Income (Annual)</Label>
                <Input
                  type="number"
                  placeholder="e.g., 30000 (optional)"
                  value={formData.coapplicant_income}
                  onChange={(e) => handleInputChange('coapplicant_income', e.target.value)}
                />
              </div>

              <div>
                <Label htmlFor="loan_amount">Loan Amount (in thousands)</Label>
                <Input
                  type="number"
                  placeholder="e.g., 150"
                  value={formData.loan_amount}
                  onChange={(e) => handleInputChange('loan_amount', e.target.value)}
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="loan_amount_term">Loan Term (days)</Label>
                  <Select onValueChange={(value) => handleInputChange('loan_amount_term', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="360" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="120">120 days</SelectItem>
                      <SelectItem value="180">180 days</SelectItem>
                      <SelectItem value="240">240 days</SelectItem>
                      <SelectItem value="360">360 days</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="credit_history">Credit History</Label>
                  <Select onValueChange={(value) => handleInputChange('credit_history', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Good" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Good</SelectItem>
                      <SelectItem value="0">Poor</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {error && (
                <Alert className="border-red-200 bg-red-50">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-600">{error}</AlertDescription>
                </Alert>
              )}

              <Button 
                type="submit" 
                className="w-full bg-blue-600 hover:bg-blue-700"
                disabled={loading}
              >
                {loading ? 'Analyzing...' : 'Predict Loan Risk'}
              </Button>
            </form>
          </CardContent>
        </Card>

        {prediction && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                {getPredictionIcon(prediction.predicted_class)}
                <span>Prediction Result</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center">
                <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${getRiskColor(prediction.risk_level)}`}>
                  {prediction.risk_level} Risk
                </div>
                <p className="text-2xl font-bold mt-2 mb-1">
                  {prediction.predicted_class}
                </p>
                <p className="text-gray-600">
                  Default Probability: {(prediction.predicted_default_probability * 100).toFixed(1)}%
                </p>
              </div>

              <div className="space-y-3">
                <h4 className="font-semibold text-gray-900">Application Summary</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-600">Applicant Income:</span>
                    <span className="ml-2 font-medium">${prediction.application.applicant_income.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Loan Amount:</span>
                    <span className="ml-2 font-medium">${(prediction.application.loan_amount * 1000).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Credit History:</span>
                    <span className="ml-2 font-medium">{prediction.application.credit_history ? 'Good' : 'Poor'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Property Area:</span>
                    <span className="ml-2 font-medium">{prediction.application.property_area}</span>
                  </div>
                </div>
              </div>

              <div className={`p-4 rounded-lg ${prediction.predicted_class === 'Approved' ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                <h4 className={`font-semibold ${prediction.predicted_class === 'Approved' ? 'text-green-800' : 'text-red-800'}`}>
                  Recommendation
                </h4>
                <p className={`text-sm mt-1 ${prediction.predicted_class === 'Approved' ? 'text-green-700' : 'text-red-700'}`}>
                  {prediction.predicted_class === 'Approved' 
                    ? 'This application shows low default risk and can be considered for approval.'
                    : 'This application shows high default risk. Consider requesting additional documentation or collateral.'}
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

// Data Analysis Component
const DataAnalysis = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/dataset/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
        <p className="text-gray-600">Failed to load data analysis</p>
      </div>
    );
  }

  const propertyAreaData = [
    { name: 'Urban', value: stats.feature_distributions.property_area_distribution.Urban },
    { name: 'Semiurban', value: stats.feature_distributions.property_area_distribution.Semiurban },
    { name: 'Rural', value: stats.feature_distributions.property_area_distribution.Rural }
  ];

  const educationData = [
    { name: 'Graduate', value: stats.feature_distributions.education_distribution.Graduate },
    { name: 'Not Graduate', value: stats.feature_distributions.education_distribution['Not Graduate'] }
  ];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Data Analysis & Insights</h2>
        <p className="text-gray-600">Comprehensive analysis of the loan dataset</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Property Area Distribution</CardTitle>
            <CardDescription>Geographic distribution of loan applications</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={propertyAreaData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Education Level Distribution</CardTitle>
            <CardDescription>Educational background of applicants</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={educationData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {educationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Dataset Overview</CardTitle>
          <CardDescription>Key statistics and insights from the loan dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Risk Factors</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Poor Credit History:</span>
                  <span className="text-sm font-medium text-red-600">
                    {((stats.feature_distributions.credit_history_distribution['0'] / stats.total_records) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Low Income Applicants:</span>
                  <span className="text-sm font-medium text-yellow-600">
                    {((stats.feature_distributions.income_distribution.low / stats.total_records) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Financial Metrics</h3>
              <div className="space-y-2">
                <div>
                  <span className="text-sm text-gray-600">Average Income:</span>
                  <p className="text-lg font-semibold text-green-600">${stats.avg_income.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Average Loan Amount:</span>
                  <p className="text-lg font-semibold text-blue-600">${(stats.avg_loan_amount * 1000).toLocaleString()}</p>
                </div>
              </div>
            </div>
            
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Data Quality</h3>
              <div className="space-y-2">
                <div>
                  <span className="text-sm text-gray-600">Total Records:</span>
                  <p className="text-lg font-semibold text-blue-600">{stats.total_records.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Missing Values:</span>
                  <p className="text-lg font-semibold text-green-600">0%</p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Main App Component
function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <BrowserRouter>
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<LoanPredictor />} />
            <Route path="/data" element={<DataAnalysis />} />
          </Routes>
        </main>
      </BrowserRouter>
    </div>
  );
}

export default App;