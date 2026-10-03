

import React, { useState } from "react";
import axios from "axios";
import { ENDPOINTS } from "./api";

// The API wants a picture (a multipart field called "image"), not the JSON this
// old demo component sends, so the request itself is still wrong - but the URL
// is now the routed one, so it reaches Flask locally (dev-server proxy) and on
// Vercel (vercel.json rewrite).
function Send() {
  const [name, setName] = useState("john");

  const handleClick = async () => {
    try {
      const response = await axios.post(
        ENDPOINTS.storeImage,
        { name },
        { headers: { "Content-Type": "application/json" } }
      );
      console.log(response.data);
    } catch (error) {
      console.error(error);
    }
  };


  return (
    <div>
      <h1>{name}</h1>
      <button onClick={handleClick}>Send Name to Backend</button>
    </div>
  );
}

export default Send;
