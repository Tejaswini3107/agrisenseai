'use client'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Navigation */}
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-primary">🌾 AgrisenseAI</h1>
            </div>
            <div className="flex space-x-4">
              <a href="/dashboard" className="btn-primary">
                Dashboard
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Agricultural Intelligence Platform
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            Real-time crop monitoring, analytics, and AI-powered insights
          </p>
          <a href="/dashboard" className="btn-primary text-lg">
            Access Dashboard →
          </a>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20">
          <div className="card text-center">
            <div className="text-4xl mb-4">📡</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Real-Time Monitoring
            </h3>
            <p className="text-gray-600">
              IoT sensors provide live data on soil moisture, temperature, and crop health
            </p>
          </div>
          <div className="card text-center">
            <div className="text-4xl mb-4">📊</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Advanced Analytics
            </h3>
            <p className="text-gray-600">
              Comprehensive dashboards and reports to optimize farming decisions
            </p>
          </div>
          <div className="card text-center">
            <div className="text-4xl mb-4">🤖</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              AI Insights
            </h3>
            <p className="text-gray-600">
              Machine learning models predict crop diseases and optimize yields
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
