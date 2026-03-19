'use client'

interface StatCardProps {
  title: string
  value: string | number
  icon?: string
  color?: 'primary' | 'success' | 'warning' | 'danger'
}

const colorMap = {
  primary: 'bg-blue-100 text-blue-900',
  success: 'bg-green-100 text-green-900',
  warning: 'bg-yellow-100 text-yellow-900',
  danger: 'bg-red-100 text-red-900',
}

export default function StatCard({ title, value, icon, color = 'primary' }: StatCardProps) {
  return (
    <div className={`card ${colorMap[color]} border-l-4 border-${color}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
        </div>
        {icon && <span className="text-4xl">{icon}</span>}
      </div>
    </div>
  )
}
