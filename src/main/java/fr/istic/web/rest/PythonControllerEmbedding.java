package fr.istic.web.rest;

import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Path("/api")
public class PythonControllerEmbedding {

    private final Logger log = LoggerFactory.getLogger(PythonControllerEmbedding.class);
    private Process pythonProcess;
    private BufferedWriter processWriter;
    private BufferedReader processReader;

    @POST
    @Path("/initialize-model")
    @Produces(MediaType.APPLICATION_JSON)
    public Response initializeModel() {
        log.info("Initializing the model...");

        try {
            // Start the Python process
            ProcessBuilder pb = new ProcessBuilder("python3", "src/main/embedding/embedding.py");
            pb.redirectErrorStream(true);
            pythonProcess = pb.start();

            // Streams to communicate with the process
            processWriter = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream()));
            processReader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));

            log.info("Model initialized and waiting for data...");
            return Response.ok(Map.of("status", "Model initialized and waiting for data.")).build();
        } catch (Exception e) {
            log.error("Error while initializing the model.", e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(Map.of("error", "Error while initializing the model: " + e.getMessage()))
                    .build();
        }
    }

    @POST
    @Path("/submit-data")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response submitData(Map<String, Object> requestData) {
        Map<String, Object> response = new HashMap<>();

        try {
            log.info("Submitting complete data...");
            Object textsObj = requestData.get("texts");
            List<String> texts = new ArrayList<>();
            if (textsObj instanceof List<?>) {
                for (Object item : (List<?>) textsObj) {
                    if (item instanceof String) {
                        texts.add((String) item);
                    } else {
                        return Response.status(Response.Status.BAD_REQUEST)
                                .entity(Map.of("error", "Invalid type for items in 'texts' list."))
                                .build();
                    }
                }
            } else {
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Invalid type for 'texts' in the request."))
                        .build();
            }
    
            if (texts.isEmpty()) {
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "No text data provided in the request."))
                        .build();
            }
    
            String textsJson = new JSONArray(texts).toString();

            // Send data to running process
            log.info("Sending texts to Python process: {}", textsJson);
            processWriter.write(textsJson);
            processWriter.newLine();
            processWriter.flush();

            // Read output from running process
            StringBuilder output = new StringBuilder();
            String line;
            int linesRead = 0;
            
            // Collect output until we find a valid JSON array
            while ((line = processReader.readLine()) != null) {
                linesRead++;
                
                // Log each line for debugging
                log.debug("Python output line {}: {}", linesRead, line);
                
                // Skip debug or error messages that start with "DEBUG:" or "ERROR:"
                if (line.startsWith("DEBUG:") || line.startsWith("ERROR:")) {
                    log.info("Python debug/error output: {}", line);
                    continue;
                }
                
                // Try to parse as JSON to see if it's valid
                try {
                    // Check if it's a JSONArray or JSONObject
                    if ((line.trim().startsWith("[") && line.trim().endsWith("]")) || 
                        (line.trim().startsWith("{") && line.trim().endsWith("}"))) {
                        output.append(line);
                        break; // Found valid JSON, stop reading
                    }
                } catch (Exception e) {
                    // Not valid JSON, continue collecting output
                    log.debug("Line is not valid JSON, continuing to next line");
                }
                
                // If we've read too many lines without finding valid JSON, break to avoid hanging
                if (linesRead > 50) {
                    log.warn("Read 50 lines without finding valid JSON output, stopping");
                    break;
                }
            }

            if (output.length() == 0) {
                log.error("No valid JSON output received from Python process");
                response.put("error", "No valid JSON output received from Python process");
                return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                        .entity(response)
                        .build();
            }

            // Extract embeddings
            List<Map<String, Object>> embeddings = parseEmbeddingsFromOutput(output.toString());

            // Build response JSON
            response.put("status", "success");
            response.put("message", "Embeddings generated successfully.");
            response.put("embeddings", embeddings);

            return Response.ok(response).build();
        } catch (Exception e) {
            log.error("Error while submitting data.", e);
            response.put("error", "Error while submitting data: " + e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(response)
                    .build();
        }
    }

    private List<Map<String, Object>> parseEmbeddingsFromOutput(String output) {
        log.debug("Parsing output: {}", output);
        List<Map<String, Object>> result = new ArrayList<>();
        
        try {
            // First, check if output is a JSON object with an error message
            if (output.startsWith("{")) {
                JSONObject jsonObj = new JSONObject(output);
                if (jsonObj.has("error")) {
                    log.error("Python script returned error: {}", jsonObj.getString("error"));
                    throw new RuntimeException("Python script error: " + jsonObj.getString("error"));
                }
            }
            
            // Parse as JSON array
            JSONArray embeddingsArray = new JSONArray(output);
            
            // Process each embedding
            for (int i = 0; i < embeddingsArray.length(); i++) {
                Object embedding = embeddingsArray.get(i);
                Map<String, Object> embeddingMap = new HashMap<>();
                
                // Make sure embedding is properly handled as a List/Array
                if (embedding instanceof JSONArray) {
                    // Convert JSONArray to List
                    embeddingMap.put("embedding", ((JSONArray) embedding).toList());
                } else if (embedding instanceof String) {
                    // If it's a string, try parsing it as a JSON array
                    try {
                        JSONArray embeddingArray = new JSONArray(embedding.toString());
                        embeddingMap.put("embedding", embeddingArray.toList());
                    } catch (JSONException e) {
                        // If parsing fails, wrap it in an array with a single element
                        log.warn("Could not parse embedding as JSON array, using as raw string");
                        List<String> singleEmbedding = new ArrayList<>();
                        singleEmbedding.add(embedding.toString());
                        embeddingMap.put("embedding", singleEmbedding);
                    }
                } else {
                    // For any other type, convert to string and wrap in a list
                    log.warn("Unexpected embedding type: {}", embedding.getClass().getName());
                    List<String> singleEmbedding = new ArrayList<>();
                    singleEmbedding.add(embedding.toString());
                    embeddingMap.put("embedding", singleEmbedding);
                }
                
                result.add(embeddingMap);
            }
        } catch (JSONException e) {
            log.error("Failed to parse JSON from Python output: {}", e.getMessage());
            throw new RuntimeException("Failed to parse JSON from Python output: " + e.getMessage(), e);
        }
        
        return result;
    }
}