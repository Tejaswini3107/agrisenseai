"use client";

import { useSession, signOut } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import StatCard from "@/components/StatCard";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://agrisenseai.duckdns.org";

export default function DashboardPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const sessionData = useSession();
  const [totalFarmers, setTotalFarmers] = useState<number | null>(null);
  const [cropHealth, setCropHealth] = useState<any[]>([]);
  const [farmerData, setFarmerData] = useState<any[]>([]);
  const [systemStatus, setSystemStatus] = useState({
    backend: false,
    weather: false,
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && sessionData?.status === "unauthenticated") {
      router.push("/login");
    }
  }, [sessionData?.status, router, mounted]);

  useEffect(() => {
    if (!mounted) return;

    fetch(`${BACKEND_URL}/api/farmers/`)
      .then((res) => res.json())
      .then((data) => {
        setTotalFarmers(data.length);

        const monthCounts: { [key: string]: number } = {};
        data.forEach((farmer: any) => {
          const month = new Date(farmer.created_at).toLocaleString("default", {
            month: "short",
            year: "numeric",
          });
          monthCounts[month] = (monthCounts[month] || 0) + 1;
        });

        const chartData = Object.entries(monthCounts).map(([month, count]) => ({
          month,
          farmers: count,
        }));
        setFarmerData(chartData);
      })
      .catch(() => {
        setTotalFarmers(0);
        setFarmerData([]);
      });

    fetch(`${BACKEND_URL}/api/crop-health`)
      .then((res) => res.json())
      .then((data) => setCropHealth(data.crops))
      .catch(() => setCropHealth([]));

    Promise.all([
      fetch(`${BACKEND_URL}/health`).then((r) => r.ok).catch(() => false),
      fetch(`${BACKEND_URL}/api/weather/health`).then((r) => r.ok).catch(() => false),
    ]).then(([backend, weather]) => {
      setSystemStatus({ backend, weather });
    });
  }, [mounted]);

  if (!mounted || sessionData?.status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-500">Loading...</p>
      </div>
    );
  }

  const getHealthColor = (status: string) => {
    if (status === "excellent" || status === "healthy") return "text-green-600";
    if (status === "good") return "text-blue-500";
    if (status === "fair") return "text-yellow-500";
    return "text-red-500";
  };

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Dashboard Overview</h2>
        <button
          onClick={() => signOut({ callbackUrl: "/login" })}
          className="bg-red-500 hover:bg-red-600 text-white text-sm px-4 py-2 rounded-lg transition"
        >
          Sign Out
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <StatCard
          title="Total Farmers"
          value={totalFarmers === null ? "..." : totalFarmers}
          icon="👨‍🌾"
          color="success"
        />
        <StatCard
          title="Backend API"
          value={systemStatus.backend ? "Online" : "Offline"}
          icon="🖥️"
          color={systemStatus.backend ? "success" : "danger"}
        />
        <StatCard
          title="Weather Service"
          value={systemStatus.weather ? "Online" : "Offline"}
          icon="🌤️"
          color={systemStatus.weather ? "success" : "danger"}
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">

        {/* Crop Health Bar Chart */}
        <div className="bg-white rounded-xl shadow-sm p-5">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">
            Crop Health Comparison
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={cropHealth}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={(value) => [`${value}%`, "Health"]} />
              <Bar dataKey="health" fill="#16a34a" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Farmer Registration Line Chart */}
        <div className="bg-white rounded-xl shadow-sm p-5">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">
            Farmer Registrations
          </h3>
          {farmerData.length === 0 ? (
            <div className="h-[250px] flex items-center justify-center text-gray-400 text-sm">
              No registration data yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={farmerData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="farmers"
                  stroke="#16a34a"
                  strokeWidth={2}
                  dot={{ fill: "#16a34a" }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

      </div>

      {/* Crop Health Cards */}
      <div className="bg-white rounded-xl shadow-sm p-5 mb-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Crop Health Scores
        </h3>
        {cropHealth.length === 0 ? (
          <p className="text-gray-400 text-sm">Loading crop data...</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {cropHealth.map((crop: any) => (
              <div key={crop.name} className="text-center p-3 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-700 capitalize">
                  {crop.name}
                </p>
                <p className={`text-2xl font-bold mt-1 ${getHealthColor(crop.status)}`}>
                  {crop.health}%
                </p>
                <p className={`text-xs mt-1 capitalize ${getHealthColor(crop.status)}`}>
                  {crop.status}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* System Status */}
      <div className="bg-white rounded-xl shadow-sm p-5">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          System Status
        </h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Backend API</span>
            <span className={`text-xs px-2 py-1 rounded-full ${systemStatus.backend ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
              {systemStatus.backend ? "Online" : "Offline"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Weather Service</span>
            <span className={`text-xs px-2 py-1 rounded-full ${systemStatus.weather ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
              {systemStatus.weather ? "Online" : "Offline"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">BharatGen AI</span>
            <span className={`text-xs px-2 py-1 rounded-full ${systemStatus.backend ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
              {systemStatus.backend ? "Active" : "Inactive"}
            </span>
          </div>
        </div>
      </div>

    </div>
  );
}