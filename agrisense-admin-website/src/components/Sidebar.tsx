'use client'

import Link from 'next/link'
import { useState } from 'react'

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false)

  const menuItems = [
    { label: 'Dashboard', href: '/dashboard', icon: '📊' },
    { label: 'Crops', href: '/dashboard/crops', icon: '🌾' },
    { label: 'Analytics', href: '/dashboard/analytics', icon: '📈' },
    { label: 'Alerts', href: '/dashboard/alerts', icon: '⚠️' },
    { label: 'Users', href: '/dashboard/users', icon: '👥' },
    { label: 'Settings', href: '/dashboard/settings', icon: '⚙️' },
  ]

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="md:hidden fixed top-4 left-4 z-40 p-2 bg-primary text-white rounded"
      >
        ☰
      </button>

      {/* Sidebar */}
      <aside
        className={`${
          isOpen ? 'block' : 'hidden'
        } md:block w-64 bg-white border-r border-gray-200 p-6 fixed md:relative h-full md:h-auto`}
      >
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-primary">🌾 AgrisenseAI</h2>
          <p className="text-sm text-gray-500">Admin Dashboard</p>
        </div>

        <nav className="space-y-2">
          {menuItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="flex items-center space-x-3 px-4 py-3 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>

        <div className="mt-8 pt-8 border-t">
          <button className="w-full px-4 py-2 text-left text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
            🚪 Logout
          </button>
        </div>
      </aside>
    </>
  )
}
