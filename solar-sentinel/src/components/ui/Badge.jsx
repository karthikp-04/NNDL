export default function Badge({ children, variant = 'outline', className = '', ...props }) {
  // variant: 'outline' (default neutral) | 'accent' | 'pass' | 'fail' | 'warn'
  const variantClass = variant === 'outline'
    ? 'badge-outline'
    : `badge-${variant}`

  return (
    <span className={`badge ${variantClass} ${className}`} {...props}>
      {children}
    </span>
  )
}
