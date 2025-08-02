const Button = ({ children, className = '', ...props }) => (
  <button
    className={`px-4 py-2 rounded-md font-semibold shadow-sm transition duration-200 hover:opacity-90 ${className}`}
    {...props}
  >
    {children}
  </button>
);
