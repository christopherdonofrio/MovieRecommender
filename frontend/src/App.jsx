import { useState, useRef } from "react";

// Set VITE_API_URL in production 
// at the deployed backend. Falls back to the local dev server otherwise.
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [message, setMessage] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const recommendationsRef = useRef(null);

  async function handleUpload() {
    if (!selectedFile) {
      setMessage("Please choose a CSV first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_URL}/recommend`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    setMessage(data.message);
    setRecommendations(data.recommendations || []);
    
    setTimeout(() => {
      recommendationsRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
      });
    }, 100);


  }

  return (
  <main className="recommender-layout">
    <div className="header">
        <h1>Letterboxd Movie Recommender</h1>

    </div>

    <div className ="container">

      <section className="instructions-panel">


     

        <ul className="instruction-list">
          <li>

            <div>
              <h2>1. Download your Letterboxd data</h2>
              <p>
                Visit your Letterboxd data settings and request an export of your
                account data.
              </p>

              <a
                href="https://letterboxd.com/settings/data/"
                target="_blank"
                rel="noreferrer"
              >
                Open Letterboxd data settings
              </a>
            </div>
          </li>

          <li>

            <div>
              <h2>2. Find the ratings CSV</h2>
              <p>
                Open the downloaded export folder and locate the file named
                <strong> ratings.csv</strong>.
              </p>
            </div>
          </li>

          <li>
            <div>
              <h2>3. Upload your file</h2>
              <p>
                Choose the CSV on the right and click Get Recommendations.
              </p>
            </div>
          </li>
        </ul>
      </section>

      <section className="upload-panel">
        <div className="upload-card">
          <h2>Upload your ratings</h2>

          <p className="upload-description">
            Select the ratings.csv file from your Letterboxd export.
          </p>

          <label className="file-upload">
            <div className="upload-icon">↑</div>

            <span className="upload-title">
              {selectedFile ? selectedFile.name : "Choose your ratings.csv"}
            </span>

            <span className="upload-subtitle">
              Click here to browse your files
            </span>

            <input
              type="file"
              accept=".csv"
              onChange={(event) => setSelectedFile(event.target.files[0])}
            />
          </label>

          <button
            className="recommend-button"
            onClick={handleUpload}
            disabled={!selectedFile}
          >
            Get Recommendations
          </button>

          {message && <p className="message">{message}</p>}
        </div>

      </section>
    </div>

        {recommendations.length > 0 && (
          <div className="recommendations" ref={recommendationsRef}>
            <h1 className = 'recHeading'>Your Top Recommendations</h1>

            <ol className="recommendation-list">
              {recommendations.map((movie, index) => (
                <li key={`${movie.title}-${index}`}>

                  <div className="movie-details">
                    <h3>{movie.title}</h3>
                    <p>Predicted rating: {Number(movie.score).toFixed(2)} / 5</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        )}

    <footer className="site-footer">
      <a
        href="https://github.com/christopherdonofrio/MovieRecommender"
        target="_blank"
        rel="noreferrer"
      >
        GitHub
      </a>
      <a
        href="https://www.linkedin.com/in/christopher-donofrio7705/"
        target="_blank"
        rel="noreferrer"
      >
        LinkedIn
      </a>
    </footer>

  </main>
);
}

export default App;