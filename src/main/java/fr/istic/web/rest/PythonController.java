
package fr.istic.web.rest;

import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * REST controller to run a Python script
 */
@Path("/api")
public class PythonController {

    private final Logger log = LoggerFactory.getLogger(PythonController.class);

    @POST
    @Path("/run-dan")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response runPythonScript(Map<String, Object> requestData) {
        Map<String, Object> response = new HashMap<>();
        StringBuilder output = new StringBuilder();
        StringBuilder errorOutput = new StringBuilder();

        try {
            log.info("Lancement du script Python...");

            // Récupérer l'argument envoyé dans la requête JSON
            String arg = requestData.containsKey("arg") ? requestData.get("arg").toString() : "";

            // Chemin absolu vers le script Python
            String scriptPath = "/home/thomas/Documents/Projet4A/corrigeExamBackDAN/src/main/resources/DAN_script/run_dan.py";

            // Afficher le répertoire de travail actuel
            String workingDir = System.getProperty("user.dir");
            log.info("Répertoire de travail actuel: " + workingDir);

            // Vérifier si le fichier existe
            File scriptFile = new File(scriptPath);
            if (!scriptFile.exists()) {
                log.error("Le fichier script Python est introuvable: " + scriptPath);
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Le fichier script Python est introuvable: " + scriptPath))
                        .build();
            }

            // Commande pour exécuter le script Python avec l'argument
            ProcessBuilder pb = new ProcessBuilder("python3", scriptFile.getAbsolutePath(), arg);
            pb.directory(scriptFile.getParentFile());  // Définir le répertoire de travail
            Process process = pb.start();

            // Lire la sortie standard du script Python
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                log.info("Sortie Python: " + line);
            }

            // Lire les erreurs éventuelles du script Python
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                errorOutput.append(errorLine).append("\n");
                log.error("Erreur/Avertissement Python: " + errorLine);
            }

            // Attendre la fin du processus
            int exitCode = process.waitFor();
            log.info("Processus terminé avec code : " + exitCode);

            // Construire la réponse JSON
            response.put("exitCode", exitCode);
            response.put("output", output.toString());

            // Si tout s'est bien passé
            if (exitCode == 0) {
                if (errorOutput.length() > 0) {
                    // Si des avertissements sont présents
                    response.put("warnings", errorOutput.toString());
                }
                return Response.ok(response).build();
            } else {
                // S'il y a des erreurs
                response.put("error", errorOutput.toString());
                return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                        .entity(response)
                        .build();
            }

        } catch (Exception e) {
            log.error("Erreur lors de l'exécution du script Python", e);
            response.put("error", "Erreur lors de l'exécution du script Python: " + e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(response)
                    .build();
        }
    }
}