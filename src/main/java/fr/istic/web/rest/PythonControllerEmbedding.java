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
            System.out.println(textsJson);
            processWriter.write(textsJson);
            processWriter.newLine();
            processWriter.flush();

            // Read output from running process
            StringBuilder output = new StringBuilder();
            String line;
            int linesRead = 0;
            while (linesRead < texts.size() && (line = processReader.readLine()) != null) {
                if (!line.startsWith("[") && !line.endsWith("]")) {
                    continue;
                }
                output.append(line).append("\n");
                log.info("Python Output: " + line);
                linesRead++;
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
        // Parse JSON output from the Python script
        JSONArray embeddingsArray = new JSONArray(output);
        return embeddingsArray.toList().stream()
                .map(embedding -> {
                    Map<String, Object> embeddingMap = new HashMap<>();
                    embeddingMap.put("embedding", embedding);
                    return embeddingMap;
                })
                .collect(Collectors.toList());
    }
}
