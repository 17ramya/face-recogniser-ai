
import { useState, useEffect } from 'react';
import { API_BASE_URL } from './api';

function App() {
  const [professor, setProfessor] = useState(null);

  useEffect(() => {
    // The Flask API, on the routed path: relative in the browser, forwarded by
    // the dev-server proxy locally and by the vercel.json rewrite when the two
    // halves are deployed as one project.
    fetch(`${API_BASE_URL}/api/people`)
      .then(response => response.json())
      .then(data => setProfessor((data.people || [])[0] || null))
      .catch(error => console.error(error));
  }, []);

  return (
    <div>
      {professor ? (
        <div>
          <h1>{professor.name}</h1>
          <p>Age: {professor.age}</p>
          <p>Department: {professor.department}</p>
        </div>
      ) : (
        <p>Loading professor details...</p>
      )}
    </div>
  );
}

export default App;
